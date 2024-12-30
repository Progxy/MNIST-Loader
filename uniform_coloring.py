import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from enum import Enum
import visualkeras
import pandas  as pd
import seaborn as sn
import numpy as np
import cv2
import shutil
import gzip
import struct

class Ranger(Enum):
  ABOVE    = 1
  UNDER    = 2
  IN_RANGE = 3

  def is_in_range(value, expected_value, range_perc, range_type):
    range_diff = np.floor((expected_value / 100) * range_perc).astype(int)
    if range_type == Ranger.ABOVE:
      return value > expected_value
    elif range_type == Ranger.UNDER:
      return value < expected_value
    elif range_type == Ranger.IN_RANGE:
      return (value >= (expected_value - range_diff)) and (value <= (expected_value + range_diff))

class Classifier:
    def __init__(self, width, height, model = None, epochs = 10, model_name = "", dataset_prefix = "", labels_mapping = {}, labels_cnt = 0, labels_diff = 0, train = False, use_validation_data = False, use_compression = False, enable_debug = False):
        self.width = width
        self.height = height
        self.epochs = epochs
        self.model = model
        self.dataset_prefix = dataset_prefix
        self.labels_mapping = labels_mapping
        self.labels_cnt = labels_cnt
        self.labels_diff = labels_diff
        self.use_validation_data = use_validation_data
        self.use_compression = use_compression
        self.enable_debug = enable_debug
        assert (model_name != "")
        if train:
          self.train_model()
          self.save_model(f"./out/{model_name}")
        pass

    def train_model(self):
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()  # Use all available GPUs
            print("Running on GPU")
        else:
            strategy = tf.distribute.get_strategy()  # Default strategy for CPU
            print("Running on CPU")

        if self.use_validation_data: print("Using validation data")

        with strategy.scope():
            # Load and preprocess the training data
            x_train, y_train = self.load_emnist(
                f"{self.dataset_prefix}-train-labels-idx1-ubyte",
                f"{self.dataset_prefix}-train-images-idx3-ubyte"
            )
            x_train = x_train.reshape(-1, self.height, self.width, 1).astype('float32') / 255.0  # Normalize and reshape
            y_train = y_train - self.labels_diff  # Adjust labels to 0 index

            x_test, y_test = self.load_emnist(
                f"{self.dataset_prefix}-test-labels-idx1-ubyte",
                f"{self.dataset_prefix}-test-images-idx3-ubyte"
            )
            x_test = x_test.reshape(-1, self.height, self.width, 1).astype('float32') / 255.0  # Normalize and reshape
            y_test = y_test - self.labels_diff  # Adjust labels to 0 index

            if self.use_validation_data:
              x_validation, y_validation = self.load_emnist(
                  f"{self.dataset_prefix}-validation-labels-idx1-ubyte",
                  f"{self.dataset_prefix}-validation-images-idx3-ubyte"
              )
              x_validation = x_validation.reshape(-1, self.height, self.width, 1).astype('float32') / 255.0  # Normalize and reshape
              y_validation = y_validation - self.labels_diff  # Adjust labels to 0 index


            self.model = Sequential([
                Input(shape=(self.height, self.width, 1)),

                # Convolutional Block 1
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),

                # Convolutional Block 2
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),

                # Convolutional Block 3
                Conv2D(256, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),

                # Convolutional Block 4
                Conv2D(512, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),

                # Global Feature Extraction
                GlobalAveragePooling2D(),

                # Fully Connected Layers
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(self.labels_cnt, activation='softmax')
            ])

            # Compile the model
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_accuracy',     # Monitor validation loss to adjust learning rate
                factor=0.5,             # Reduce learning rate by half each time
                patience=3,             # Wait 3 epochs for improvement before reducing the learning rate
                min_lr=1e-6,            # Set a very low minimum learning rate
                verbose=1               # Print updates when the learning rate is reduced
            )

            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )

            # Prepare the data pipeline
            batch_size = 128 if isinstance(strategy, tf.distribute.TPUStrategy) else 64  # Adjust batch size for TPU

            if self.use_validation_data:
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
                    .shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, y_validation)) \
                    .batch(batch_size).prefetch(tf.data.AUTOTUNE)
                self.model.fit(
                    train_dataset,
                    epochs=self.epochs,
                    validation_data=validation_dataset,
                    callbacks=[reduce_lr, early_stop],
                    verbose=2
                )

            else:
              self.model.fit(
                    x_train, y_train,
                    epochs=self.epochs,
                    validation_split=0.1,
                    callbacks=[reduce_lr, early_stop],
                    verbose=2
                )

            self.evaluate_model(x_test, y_test)

            return

    def evaluate_model(self, x_test, y_test):
        # Evaluate the model on test data
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test)
        print(f'Loss on test set: {test_loss:.4f}')
        print(f'Accuracy on test set: {test_accuracy:.4f}')

        labels_names = list(self.labels_mapping.values())
        y_pred = self.model.predict(x_test)
        matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
        df_cm = pd.DataFrame(matrix, labels_names, labels_names)
        plt.figure(figsize = (10,7))
        sn.set_theme(font_scale=1.4) #for label size
        sn.heatmap(df_cm, cmap="BuPu", annot=True, annot_kws={"size": self.labels_cnt}) #font size
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.title('Confusion Matrix')
        plt.savefig("./out/confusion_matrix.png")
        plt.show()

        return

    def visualize_model(self):
      visualkeras.layered_view(self.model, legend_text_spacing_offset=0).show() # display using your system viewer
      visualkeras.layered_view(self.model, legend_text_spacing_offset=0, to_file='./out/model.png') # write to disk
      return


    def save_model(self, model_name):
      self.model.save(model_name)
      return

    # Classify a single processed image
    def classify_image(self, image):
        if self.model == None: self.train_model()

        # Load and preprocess the image
        image = cv2.resize(image, (self.height, self.width)).astype('float32')  # Resize to 28x28
        if self.enable_debug:
          print("Resized Image:")
          cv2.imshow(image)
        image = image.reshape(1, self.height, self.width, 1)  # Reshape to model input shape

        # Predict the class
        prediction = self.model.predict(image)
        predicted_label = np.argmax(prediction)
        predicted_char = self.labels_mapping[predicted_label]  # Map label to character
        print(f"Predicted Label: {predicted_label}, Predicted Character: {predicted_char}")
        return

    def load_emnist(self, label_path, image_path):
        if self.use_compression:
            label_path += ".gz"
            image_path += ".gz"

        labels = []
        labels_data = b""
        with open(label_path, 'rb') as f:
            labels_data = f.read()

        if self.use_compression: labels_data = gzip.decompress(labels_data)
        magic, size = struct.unpack('>II', labels_data[:8])
        assert (magic == 2049)
        print(f"file_name: {label_path}, magic: {magic}, size: {size}")
        labels = np.frombuffer(labels_data[8:], dtype=np.uint8)

        images = []
        images_data = b""
        with open(image_path, 'rb') as f:
            images_data = f.read()

        if self.use_compression: images_data = gzip.decompress(images_data)
        magic, size, rows, cols = struct.unpack('>IIII', images_data[:16])
        assert (magic == 2051)
        assert (self.height == rows and self.width == cols)
        print(f"file_name: {image_path}, magic: {magic}, size: {size}, rows: {rows}, cols: {cols}")
        num_pixels = size * rows * cols
        image_data = np.frombuffer(images_data[16:16+num_pixels], dtype=np.uint8)
        images = image_data.reshape((size, rows, cols))

        return images, labels

class Recognizer:
  def __init__(self, img_path = "", img = None, range_type = Ranger.IN_RANGE, enable_debug = False):
    if img_path != "":
      self.img = cv2.imread(img_path)
    else: self.img = img
    self.range_type = range_type
    self.enable_debug = enable_debug
    self.grouped_rects = []
    self.discarded = []
    pass

  def is_near(self, rect_a, rect_b):
    contour_a, img_a, x_a, y_a, w_a, h_a = rect_a
    contour_b, img_b, x_b, y_b, w_b, h_b = rect_b
    v_diff = -((y_a + h_a) - y_b) if w_a > w_b else -((y_b - h_a) - y_a)

    if w_a > w_b:
      if not (x_a < x_b and (x_a + w_a) > (x_b + w_b)): return False
    else:
      if not (x_b < x_a and (x_b + w_b) > (x_a + w_a)): return False

    if not (v_diff >= 0 and v_diff <= (max(h_a, h_b) / 2)): return False

    return True

  def reassemble_rects(self, img, rects):
    for i in range(0, len(rects)):
      for j in range(i + 1, len(rects)):
        if self.is_near(rects[j], rects[i]):
          # Merge contours
          merged_contour = np.vstack((rects[i][0], rects[j][0]))
          x, y, w, h = cv2.boundingRect(merged_contour)
          rects.pop(j)
          rects[i] = (merged_contour, img.copy()[y:y + h, x:x + w], x, y, w, h)
          break
    return rects

  def find_rects(self, expected_w_perc, expected_h_perc, expected_diff_perc):
    gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Preprocess the image using Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if self.enable_debug:
      print("Closing Image:")
      cv2.imshow(closing)

    # Detect contours
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Determine expected size
    min_expected_h = np.ceil(expected_h_perc * (np.array(closing).shape[0] / 100)).astype(int)
    min_expected_w = np.ceil(expected_w_perc * (np.array(closing).shape[1] / 100)).astype(int)
    if self.range_type == Ranger.ABOVE:
      print(f"Expected size: ({min_expected_h}>) x ({min_expected_w}>)")
    elif self.range_type == Ranger.UNDER:
      print(f"Expected size: ({min_expected_h}<) x ({min_expected_w}<)")
    elif self.range_type == Ranger.IN_RANGE:
      h_range_diff = np.floor((min_expected_h / 100) * expected_diff_perc).astype(int)
      w_range_diff = np.floor((min_expected_w / 100) * expected_diff_perc).astype(int)
      if self.enable_debug: print(f"min_expected_h: {min_expected_h}, min_expected_w : {min_expected_w}")
      print(f"Expected size: ({min_expected_h - h_range_diff} - {min_expected_h + h_range_diff}) x ({min_expected_w - w_range_diff} - {min_expected_w + w_range_diff})")

    rects = []
    idx = 0
    discarded_idx = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if Ranger.is_in_range(w, min_expected_w, expected_diff_perc, self.range_type) and Ranger.is_in_range(h, min_expected_h, expected_diff_perc, self.range_type):
          if self.enable_debug: print(f"Rect match: {h} x {w} at {y} x {x}, index: {idx}")
          rects.append((contour, closing.copy()[y:y + h, x:x + w], x, y, w, h))
          idx += 1
        else:
          if self.enable_debug: print(f"Rect found: {h} x {w} at {y} x {x}, index: {discarded_idx}")
          self.discarded.append((closing.copy()[y:y + h, x:x + w], x, y, w, h))
          discarded_idx += 1

    rects = self.reassemble_rects(closing, rects)
    rects_cnt = len(rects)
    rects_idx = 0

    print(" -- Check for Written Sign -- ")
    while rects_idx != rects_cnt:
      contour, image, x, y, w, h = rects[rects_idx]
      erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
      eroded = cv2.erode(image.copy(), erosion_kernel, iterations = 1)

      if self.enable_debug:
        print(f"Testing Rect {rects_idx}")
        cv2.imshow(image)

      if not (self.is_a_written_sign(image, 2.8) or self.is_a_written_sign(eroded, 2.8)):
        print(f" -- Removing Rect {rects_idx} -- ")
        print("Original Image:")
        self.is_a_written_sign(image, 2.8)
        cv2.imshow(image)
        print("After Erosion:")
        self.is_a_written_sign(eroded, 2.8)
        cv2.imshow(eroded)
        print(" -- End Rect Removed -- ")
        cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        rects.pop(rects_idx)
        rects_cnt -= 1
        rects_idx -= 1

      rects_idx += 1
      if self.enable_debug: print("-------------------------")

    print("-----------------------------")

    # Sort the Cells and determine the number of elements per row and per column
    self.grouped_rects = self.group_rects(rects)

    return

  def get_matrix_list_shape(self, matrix_list):
    return len(matrix_list), len(matrix_list[0])

  def get_row_y_pos(self, row):
    contour, img, x, y, w, h = row[0]
    return y

  def get_rect_x_pos(self, rect):
    contour, img, x, y, w, h = rect
    return x

  def group_rects(self, rects):
    grouped_rects = []
    for idx_rect, rect in enumerate(rects):
      contour, img, x, y, w, h = rect
      new_row = True
      for idx, row in enumerate(grouped_rects):
        contour_1, img_1, x_1, y_1, w_1, h_1 = row[0]
        v_diff = abs(y - y_1)
        if v_diff < max(h, h_1):
          grouped_rects[idx].append(rect)
          new_row = False
          break
      if new_row:
        row = [rect]
        grouped_rects.append(row)

    grouped_rects = sorted(grouped_rects, key=self.get_row_y_pos)

    for idx, row in enumerate(grouped_rects):
      grouped_rects[idx] = sorted(row, key=self.get_rect_x_pos)

    return grouped_rects

  def is_a_written_sign(self, img, threshold = 0.0):
    total_cnt = 0
    white_cnt = 0
    black_cnt = 0

    for row in img:
      for element in row:
        if element == 255: white_cnt += 1
        else: black_cnt += 1
        total_cnt += 1

    if white_cnt == 0 or black_cnt == 0:
      if self.enable_debug: print(f"total_cnt: {total_cnt}, black_cnt: {black_cnt}, white_cnt: {white_cnt}")
      return False

    if self.enable_debug: print(f"total_cnt: {total_cnt}, black_cnt: {black_cnt}, white_cnt: {white_cnt}, percentage_a: {black_cnt/white_cnt}")

    # TODO: Apply more complex and finer strategy to determine if it is noise, a random line or a written sign like a character
    return (black_cnt/white_cnt > threshold)

  def show_result(self):
      print("\nResult:")
      (rows, cols) = self.get_matrix_list_shape(self.grouped_rects)
      print("-----------------------------------")
      print(f"Matrix shape: ({rows}, {cols})")
      print("-----------------------------------")

      for idx_row, row in enumerate(self.grouped_rects):
        print(f"Row {idx_row}:")
        for idx, cell in enumerate(row):
          contour, img, x, y, w, h = cell
          cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
          print(f"Element {idx}:")
          cv2.imshow(img)
          classifier.classify_image(img)
        print("-----------------------------------")

      for idx, discard in enumerate(self.discarded):
        img, x, y, w, h = discard
        cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2) # Add the discarded ones to the image contours
        if self.enable_debug:
            print(f"Discarded {idx}:")
            cv2.imshow(img)

      print(" -- Image with Contours -- ")
      cv2.imshow(self.img)
      print("-----------------------------------")

      return

if __name__ == "__main__":
  enable_debug = False

  labels_mapping = {0: "g", 1: "y", 2: "b", 3: "t", 4: "invalid"}
  labels_cnt = len(labels_mapping)
  # Load model from file
  #my_model = load_model("./out/my-small-model.keras")

  classifier = Classifier(width=32, height=32, model=None, epochs=50, model_name="my-small-model.keras", dataset_prefix="../../dataset_big/my-dataset", labels_mapping=labels_mapping, labels_cnt=labels_cnt, train=True, use_validation_data=True, use_compression=True, enable_debug=enable_debug)
  classifier.visualize_model()
  shutil.make_archive("out.zip", 'zip', "./out")

  #                                                        <path to image>  - Percentages -   Range Type   -   Debug?
  recognizer = Recognizer(img_path="TestImage_boh.png", range_type=Ranger.IN_RANGE, enable_debug=enable_debug)
  recognizer.find_rects(7, 7, 100)
  recognizer.show_result()