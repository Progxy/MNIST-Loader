import cv2
import numpy as np
import random
import struct
import matplotlib.pyplot as plt
import os
import gzip

class Transformations:
    def __init__(self, images): 
        self.images = images
        pass

    def apply_scaling(self, scales=[0.8, 1.0, 1.2]):
        for scale in scales:
            for img in self.images[:]:
                height, width = img.shape[:2]
                scaled = cv2.resize(img.copy(), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                image = cv2.resize(scaled, (width, height))
                self.images.append(image)

    def apply_translation(self, translations=[(-10, -10), (10, 10), (0, 20)]):
        for tx, ty in translations:
            for img in self.images[:]:
                height, width = img.shape[:2]
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                translated = cv2.warpAffine(img.copy(), M, (width, height))
                self.images.append(translated)

    def apply_shearing(self, shear_factors=[-0.2, 0.0, 0.2]):
        for shear in shear_factors:
            for img in self.images[:]:
                height, width = img.shape[:2]
                M = np.float32([[1, shear, 0], [0, 1, 0]])
                sheared = cv2.warpAffine(img.copy(), M, (width, height))
                self.images.append(sheared)

    def apply_rotation(self, angles=[-10, -5, 0, 5, 10]):
        for angle in angles:
            for img in self.images[:]:
                height, width = img.shape[:2]
                center = (width // 2, height // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img.copy(), M, (width, height))
                self.images.append(rotated)

    def apply_flipping(self):
        for img in self.images[:]:
            self.images.append(cv2.flip(img.copy(), 1))  # Horizontal flip
            self.images.append(cv2.flip(img.copy(), 0))  # Vertical flip

    def apply_contrast_adjustment(self, contrast_factors=[0.8, 1.0, 1.2]):
        for alpha in contrast_factors:
            for img in self.images[:]:
                adjusted = cv2.convertScaleAbs(img.copy(), alpha=alpha, beta=0)
                self.images.append(adjusted)

    def apply_gaussian_noise(self, noise_levels=[10, 20, 30]):
        for noise_level in noise_levels:
            for img in self.images[:]:
                noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
                noisy_image = cv2.add(img.copy(), noise)
                self.images.append(noisy_image)

    def apply_all_transformations(self, extended_transformations):
        self.apply_scaling()
        self.apply_rotation()
        self.apply_contrast_adjustment()
        if extended_transformations:
            self.apply_flipping()
            self.apply_gaussian_noise()
        return
    
    def noise_transformations(self, extended_transformation):
        if extended_transformation:
            self.apply_scaling()
        self.apply_translation()
        if extended_transformation:
            self.apply_shearing()
        self.apply_rotation()
        return

    def get_images(self):
        return self.images

class DatasetGenerator:
    def __init__(self, dataset_prefix, images_paths_and_labels = {}, width = 28, height = 28, noise_cnt = 10, limit_size = 0, use_compression = False, extended_dataset = False, balance_dataset = False, read_only=False):
        self.dataset_prefix = dataset_prefix
        assert (self.dataset_prefix != "")
        
        self.use_compression = use_compression
        self.labels = list(images_paths_and_labels.keys())
        
        if read_only: pass

        if limit_size == 0: self.limit_size = (1 << 32)
        else: self.limit_size = limit_size
        self.balance_dataset = balance_dataset

        self.dataset = {}
        self.width = width
        self.height = height
        self.extended_dataset = extended_dataset
        self.images_paths_and_labels = images_paths_and_labels
        self.sets_names = []
        self.sets = []
        self.splitting_info = []
        self.is_extended = False

        self.images = {}
        self.read_images()
        
        self.noise_images = []
        for _ in range(noise_cnt):
            self.noise_images.append(self.generate_random_noise())

        pass
    
    def read_images(self):
        for label in self.images_paths_and_labels:
            images = []
            for image_path in self.images_paths_and_labels[label]:
                image = cv2.imread(image_path)
                image = cv2.resize(image, (self.width, self.height))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                images.append(binary)
            self.images[label] = images
        return
    
    def generate_random_noise(self):
        # Create a blank canvas
        canvas = np.zeros((self.height, self.width), dtype=np.uint8)

        # Number of shapes to draw
        num_shapes = random.randint(1, 15)

        for _ in range(num_shapes):
            shape_type = random.choice(['line', 'circle', 'rectangle', 'polyline'])

            if shape_type == 'line':
                # Random line with varying thickness
                pt1 = (random.randint(0, self.width), random.randint(0, self.height))
                pt2 = (random.randint(0, self.width), random.randint(0, self.height))
                thickness = random.randint(1, 3)
                color = random.randint(100, 255)  # Brightness for grayscale
                cv2.line(canvas, pt1, pt2, color, thickness)

            elif shape_type == 'circle':
                # Random circle
                center = (random.randint(0, self.width), random.randint(0, self.height))
                radius = random.randint(5, 30)
                thickness = random.choice([-1, random.randint(1, 3)])  # -1 for filled
                color = random.randint(100, 255)
                cv2.circle(canvas, center, radius, color, thickness)

            elif shape_type == 'rectangle':
                # Random rectangle
                pt1 = (random.randint(0, self.width), random.randint(0, self.height))
                pt2 = (random.randint(0, self.width), random.randint(0, self.height))
                thickness = random.choice([-1, random.randint(1, 3)])
                color = random.randint(100, 255)
                cv2.rectangle(canvas, pt1, pt2, color, thickness)

            elif shape_type == 'polyline':
                # Random polyline
                num_points = random.randint(3, 6)
                points = np.array([
                    (random.randint(0, self.width), random.randint(0, self.height)) for _ in range(num_points)
                ], dtype=np.int32)
                color = random.randint(100, 255)
                thickness = random.randint(1, 3)
                is_closed = random.choice([True, False])
                cv2.polylines(canvas, [points], is_closed, color, thickness)

        # Add random noise
        noise = np.random.randint(0, 50, (self.height, self.width), dtype=np.uint8)
        canvas = cv2.add(canvas, noise)
        _, canvas = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return canvas
    
    def get_dataset_size(self, debug = False):
        size = 0
        for label in self.labels:
            size += len(self.dataset[label])
            if debug: print(f"label {label} contains {len(self.dataset[label])} images.")
        if debug: print(f"The dataset contains {size} images in total.")
        return size

    def generate_dataset(self):
        for label in self.images:
            print(f"Generating images of label {label}...")
            img_transformations = Transformations(self.images[label])
            img_transformations.apply_all_transformations(self.extended_dataset)
            self.dataset[label] = img_transformations.get_images()
            print(f"Successfully generated {len(self.dataset[label])} images.")

        dataset_size = self.get_dataset_size()
        if self.limit_size < dataset_size: 
            print(f"Resizing dataset from {dataset_size} to {self.limit_size}...")
        
        label_cnt = len(self.labels)
        for label in self.labels:
            label_len = len(self.dataset[label])
            min_size = self.limit_size // label_cnt

            label_cnt -= 1
            self.limit_size -= min(label_len, min_size)

            if label_len <= min_size: continue
            original_element_cnt = len(self.images_paths_and_labels[label])
            pop_indices = random.sample(range(original_element_cnt, label_len), label_len - min_size)
            pop_indices.sort(reverse=True)  # Sort indices in descending order
            for pop_idx in pop_indices:
                self.dataset[label].pop(pop_idx)
        
        if self.limit_size < dataset_size: 
            dataset_size = self.get_dataset_size()
            print(f"new dataset_size: {dataset_size}")
            for label in self.labels:
                print(f"label {label} len: {len(self.dataset[label])}")

        return
    
    def split_dataset(self, training_split = 75, validation_split = 0, test_split = 25):
        assert(training_split + validation_split + test_split == 100)

        print("Balancing dataset...")
        min_label_len = len(self.dataset[min(self.labels, key=lambda label: len(self.dataset[label]))])
        print(f"Balancing the labels to {min_label_len}...")

        for label in self.labels:
            label_len = len(self.dataset[label])
            original_element_cnt = len(self.images_paths_and_labels[label])
            if label_len == min_label_len: continue
            pop_indices = random.sample(range(original_element_cnt, label_len), label_len - min_label_len)
            pop_indices.sort(reverse=True)  # Sort indices in descending order
            for pop_idx in pop_indices:
                self.dataset[label].pop(pop_idx)

        print("Balanced all the labels")

        print(f"Splitting dataset in training set: {training_split}%, validation split: {validation_split}% and test split: {test_split}%...")
        dataset_size = len(self.dataset[0])

        if training_split > 0:
            training_set = {index: [] for index in self.labels}
            train_size = (dataset_size // 100) * training_split
            for label in self.labels:
                label_len = len(self.dataset[label])
                pop_indices = random.sample(range(0, label_len), train_size)
                pop_indices.sort(reverse=True)  # Sort indices in descending order
                for pop_idx in pop_indices:
                    training_set[label].append(self.dataset[label].pop(pop_idx))
            self.sets_names.append("train")
            self.sets.append(training_set)
            splitting_info = {'size': train_size, 'size_perc': training_split}
            self.splitting_info.append(splitting_info)
            print(f"Created training split containing {train_size} images for each label.")

        if validation_split > 0:
            validation_set = {index: [] for index in self.labels}
            validation_size = (dataset_size // 100) * validation_split
            for label in self.labels:
                label_len = len(self.dataset[label])
                pop_indices = random.sample(range(0, label_len), validation_size)
                pop_indices.sort(reverse=True)  # Sort indices in descending order
                for pop_idx in pop_indices:
                    validation_set[label].append(self.dataset[label].pop(pop_idx))
            self.sets_names.append("validation")
            self.sets.append(validation_set)
            splitting_info = {'size': validation_size, 'size_perc': validation_split}
            self.splitting_info.append(splitting_info)
            print(f"Created validation split containing {validation_size} images for each label.")

        if test_split > 0:
            test_set = {index: [] for index in self.labels}
            for label in self.labels:
                test_set[label] = self.dataset[label]
            self.sets_names.append("test")
            self.sets.append(test_set)
            test_size = len(test_set[0])
            splitting_info = {'size': test_size, 'size_perc': test_split}
            self.splitting_info.append(splitting_info)
            print(f"Created test split containing {test_size} images for each label.")

        return
    
    def generate_info_txt(self, file_name = "dataset_info.txt"):
        print(f"Generating dataset info: {file_name}...")
        with open(file_name, "w") as file:
            file.write("Dataset Info: \n")
            file.write(f" - Extended dataset: {self.is_extended}.\n")
            file.write(f" - Extended data augmentation: {self.extended_dataset}.\n")
            file.write(f" - Sets:\n")
            for idx, set_name in enumerate(self.sets_names):
                file.write(" ------------------------------------------------------------------------------------- \n")
                file.write(f" \t- Set '{set_name}':\n")
                file.write(f" \t- Labels Cnt: {len(self.labels)}.\n")
                file.write(f" \t- Size: {self.splitting_info[idx]['size']} images for each label.\n")
                file.write(f" \t- Size: {self.splitting_info[idx]['size_perc']}% of the total size of the dataset.\n")
                file.write(f" \t- Images with size {self.height} x {self.width}.\n")
                file.write(" ------------------------------------------------------------------------------------- \n")
        print("File generated successfully.")
        return
    
    def store_dataset_as_mnist_format(self):
        assert (self.dataset != {})
        for idx, set_data in enumerate(self.sets):
            labels_file_name = f"{self.dataset_prefix}-{self.sets_names[idx]}-labels-idx1-ubyte"
            if self.use_compression: labels_file_name += ".gz"
            labels_magic = 2049
            
            size = 0
            for label in self.labels:
                size += len(set_data[label])

            print(f"Generating file {labels_file_name} with {size} labels...")
            labels_data = b""
            labels_data += struct.pack(">I", labels_magic)
            labels_data += struct.pack(">I", size)
            for label in self.labels:
                labels_data += struct.pack(">B", label) * len(set_data[label])
            
            print("Writing data to file...")
            
            if self.use_compression: 
                with gzip.open(labels_file_name, 'wb', compresslevel=6) as f:
                    f.write(labels_data)
            else:
                with open(labels_file_name, "wb") as f:
                    f.write(labels_data)

            print("File generated successfully.")
            
            images_file_name = f"{self.dataset_prefix}-{self.sets_names[idx]}-images-idx3-ubyte"
            if self.use_compression: images_file_name += ".gz"
            images_magic = 2051

            print(f"Generating file {images_file_name} with {size} images...")

            images_data = b''
            images_data += struct.pack(">I", images_magic)
            images_data += struct.pack(">I", size)
            images_data += struct.pack(">I", self.height)
            images_data += struct.pack(">I", self.width)
            for label in self.labels:
                images_data += b''.join(image.tobytes() for image in set_data[label])

            print("Writing data to file...")

            if self.use_compression: 
                with gzip.open(images_file_name, 'wb', compresslevel=6) as f:
                    f.write(images_data)
            else:
                with open(images_file_name, "wb") as f:
                    f.write(images_data)

            print("File generated successfully.")

        return
    
    def extend_dataset(self, labels_file_prefix, images_file_prefix, label_mappings):
        extension_images, extension_labels = self.load_mnist_format_dataset(labels_file_prefix=labels_file_prefix, images_file_prefix=images_file_prefix)
        for idx, label in enumerate(extension_labels):
            if label_mappings[label] > 0:
                self.dataset[label_mappings[label]].append(cv2.resize(extension_images[idx], (self.width, self.height)))
        print("Dataset Extended")
        self.get_dataset_size(debug=True)
        self.is_extended = True
        return

    def load_mnist_format_dataset(self, labels_file_prefix = "", images_file_prefix = ""):
        if labels_file_prefix == "" and images_file_prefix == "":
            labels_file_prefix, images_file_prefix = f"{self.dataset_prefix}-labels", f"{self.dataset_prefix}-images"

        label_path = f"{labels_file_prefix}-idx1-ubyte"
        image_path = f"{images_file_prefix}-idx3-ubyte"
        if self.use_compression: 
            label_path += ".gz"
            image_path += ".gz"

        labels = []
        labels_data = b""
        with open(label_path, 'rb') as f:
            labels_data = f.read()

        if self.use_compression: labels_data = gzip.decompress(labels_data)
        magic, labels_size = struct.unpack('>II', labels_data[:8])
        assert (magic == 2049)
        print(f"magic: {magic}, labels_size: {labels_size}")
        labels = np.frombuffer(labels_data[8:], dtype=np.uint8).tolist()

        images = []
        images_data = b""
        with open(image_path, 'rb') as f:
            images_data = f.read()

        if self.use_compression: images_data = gzip.decompress(images_data)
        magic, images_size, rows, cols = struct.unpack('>IIII', images_data[:16])
        print(f"magic: {magic}, images_size: {images_size}, rows: {rows}, cols: {cols}")
        assert (magic == 2051 and images_size == labels_size)
        num_pixels = images_size * rows * cols
        image_data = np.frombuffer(images_data[16:16+num_pixels], dtype=np.uint8)
        images = image_data.reshape((images_size, rows, cols))
            
        return images, labels

    def show_dataset(self, x_train, y_train, mapping):
        for idx, image_data in enumerate(x_train):
            # Display the image
            plt.imshow(image_data, cmap="gray", interpolation="nearest")
            plt.axis("off")  # Turn off axis
            plt.title(f"Image {idx} with label {y_train[idx]} - {mapping[y_train[idx]]}")
            plt.show()
        return

def find_dataset_images(folder_path, exclude_substrings=[]):
    print(f"Exluding substrings: {exclude_substrings}")
    paths_and_labels = {}

    def should_exclude(file_name):
        # Check if any substring in exclude_substrings is present in the file_name
        return any(substring in file_name for substring in exclude_substrings)

    paths_and_labels[0] = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith('.png') and f.startswith('g') and not should_exclude(f)
    ]
    
    paths_and_labels[1] = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith('.png') and f.startswith('y') and not should_exclude(f)
    ]
    
    paths_and_labels[2] = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith('.png') and f.startswith('b') and not should_exclude(f)
    ]
    
    paths_and_labels[3] = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.endswith('.png') and f.startswith('t') and not should_exclude(f)
    ]

    return paths_and_labels

if __name__ == "__main__":
    generate = True # Change between generating and displaying
    if generate:
        #        --   validation test --      test set      --                                  train set                 --
        substrings = ["bpen", "pencil", "boh.png", "vpencil", "bohpenn", "hpencil", "blpen", "vpen", "g_", "y_", "b_", "t_"]
        width = 32
        height = 32

        # Find all the paths except the ones used for the test dataset
        paths_and_labels = find_dataset_images("./example", exclude_substrings=[])

        dataset_generator = DatasetGenerator(dataset_prefix="./dataset/my-dataset", images_paths_and_labels=paths_and_labels, width=width, height=height, limit_size=0, use_compression=True, extended_dataset=False, balance_dataset = True)
        # Generate multiple transformations of the given images, effectively populating the dataset (which at this point is a dictionary)
        dataset_generator.generate_dataset()
        
        # Extend dataset with the emnist letters dataset
        emnist_labels = [-1] * 62
        emnist_labels[11] = 2
        emnist_labels[16] = 0
        emnist_labels[29] = 3
        emnist_labels[34] = 1
        dataset_generator.extend_dataset(labels_file_prefix="./dataset/emnist-byclass-train-labels", images_file_prefix="./dataset/emnist-byclass-train-images", label_mappings=emnist_labels)
        dataset_generator.extend_dataset(labels_file_prefix="./dataset/emnist-byclass-test-labels", images_file_prefix="./dataset/emnist-byclass-test-images", label_mappings=emnist_labels)

        # Save the dataset using the mnist format
        dataset_generator.split_dataset(training_split=75, validation_split=10, test_split=15)
        dataset_generator.store_dataset_as_mnist_format()
        dataset_generator.generate_info_txt(file_name="./dataset/my-dataset-dataset_info.txt")

    else:
        # To load the dataset specify the prefixes as above
        dataset_viewer = DatasetGenerator(dataset_prefix="./dataset/my-dataset-validation", use_compression=True, read_only=True)
        #dataset_viewer = DatasetGenerator(dataset_prefix="./dataset/emnist-byclass-train", use_compression=True, read_only=True)
        # load the labels and the images
        x_train, y_train = dataset_viewer.load_mnist_format_dataset()
        # Plot every image inside x_train

        labels = ['g', 'y', 'b', 't']
        #labels = [chr(48), chr(49), chr(50), chr(51), chr(52), chr(53), chr(54), chr(55), chr(56), chr(57), chr(65), chr(66), chr(67), chr(68), chr(69), chr(70), chr(71), chr(72), chr(73), chr(74), chr(75), chr(76), chr(77), chr(78), chr(79), chr(80), chr(81), chr(82), chr(83), chr(84), chr(85), chr(86), chr(87), chr(88), chr(89), chr(90), chr(97), chr(98), chr(99), chr(100), chr(101), chr(102), chr(103), chr(104), chr(105), chr(106), chr(107), chr(108), chr(109), chr(110), chr(111), chr(112), chr(113), chr(114), chr(115), chr(116), chr(117), chr(118), chr(119), chr(120), chr(121), chr(122)]
        dataset_viewer.show_dataset(x_train, y_train, labels)
