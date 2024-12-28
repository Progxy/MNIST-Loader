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
                scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                image = cv2.resize(scaled, (width, height))
                self.images.append(image)

    def apply_translation(self, translations=[(-10, -10), (10, 10), (0, 20)]):
        for tx, ty in translations:
            for img in self.images[:]:
                height, width = img.shape[:2]
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                translated = cv2.warpAffine(img, M, (width, height))
                self.images.append(translated)

    def apply_shearing(self, shear_factors=[-0.2, 0.0, 0.2]):
        for shear in shear_factors:
            for img in self.images[:]:
                height, width = img.shape[:2]
                M = np.float32([[1, shear, 0], [0, 1, 0]])
                sheared = cv2.warpAffine(img, M, (width, height))
                self.images.append(sheared)

    def apply_rotation(self, angles=[-15, -10, 0, 10, 15]):
        for angle in angles:
            for img in self.images[:]:
                height, width = img.shape[:2]
                center = (width // 2, height // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (width, height))
                self.images.append(rotated)

    def apply_flipping(self):
        for img in self.images[:]:
            self.images.append(cv2.flip(img, 1))  # Horizontal flip
            self.images.append(cv2.flip(img, 0))  # Vertical flip

    def apply_contrast_adjustment(self, contrast_factors=[0.8, 1.2]):
        for alpha in contrast_factors:
            for img in self.images[:]:
                adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
                self.images.append(adjusted)

    def apply_gaussian_noise(self, noise_levels=[10, 20, 30]):
        for noise_level in noise_levels:
            for img in self.images[:]:
                noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
                noisy_image = cv2.add(img, noise)
                self.images.append(noisy_image)

    def apply_all_transformations(self):
        print("applying scaling...")
        self.apply_scaling()
        print("applying rotation...")
        self.apply_rotation()
        print("applying flipping...")
        self.apply_flipping()
        print("applying contrast adjustment...")
        self.apply_contrast_adjustment()
        print("applying gaussian noise...")
        self.apply_gaussian_noise()
        return
    
    def noise_transformations(self):
        print("applying scaling...")
        self.apply_scaling()
        print("applying translation...")
        self.apply_translation()
        print("applying shearing...")
        self.apply_shearing()
        print("applying rotation...")
        self.apply_rotation()
        return

    def get_images(self):
        return self.images

class DatasetGenerator:
    def __init__(self, labels_file_prefix, images_file_prefix, images_paths_and_labels = {}, width = 28, height = 28, noise_cnt = 10, use_compression = False):
        self.labels_file_prefix = labels_file_prefix
        assert (self.labels_file_prefix != "")
        self.images_file_prefix = images_file_prefix
        assert (self.images_file_prefix != "")

        self.dataset = {}
        self.width = width
        self.height = height
        self.use_compression = use_compression
        self.labels = list(images_paths_and_labels.keys())

        self.images = {}
        self.read_images(images_paths_and_labels)
        
        self.noise_images = []
        for _ in range(noise_cnt):
            self.noise_images.append(self.generate_random_noise())

        pass
    
    def read_images(self, images_paths_and_labels):
        for label in images_paths_and_labels:
            images = []
            for image_path in images_paths_and_labels[label]:
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
    
    def generate_dataset(self):
        for label in self.images:
            print(f"Generating images of label {label}")
            self.dataset[label] = []
            for idx, img in enumerate(self.images[label]):
                print(f"Generating images from transformation of image {idx}...")
                img_transformations = Transformations([img])
                img_transformations.apply_all_transformations()
                self.dataset[label] += img_transformations.get_images()
            print(f"Successfully generated {len(self.dataset[label])} images.")
        print(f"Generating images from transformation of noise images...")
        noise_img_transformations = Transformations(self.noise_images)
        noise_img_transformations.noise_transformations()
        self.dataset[len(self.labels)] = noise_img_transformations.get_images()
        print(f"Successfully generated {len(self.dataset[len(self.labels)])} images.")
        self.labels.append(len(self.labels))
        return
    
    def store_dataset_as_mnist_format(self):
        assert (self.dataset != {})

        labels_file_name = f"{self.labels_file_prefix}-idx1-ubyte"
        if self.use_compression: labels_file_name += ".gz"
        labels_magic = 2049
        
        size = 0
        for label in self.labels:
            size += len(self.dataset[label])

        print(f"Generating file {labels_file_name} with {size} labels...")
        labels_data = b""
        labels_data += struct.pack(">I", labels_magic)
        labels_data += struct.pack(">I", size)
        for label in self.labels:
            labels_data += struct.pack(">B", label) * len(self.dataset[label])
        
        print("Writing data to file...")
        
        if self.use_compression: 
            with gzip.open(labels_file_name, 'wb', compresslevel=6) as f:
                f.write(labels_data)
        else:
            with open(labels_file_name, "wb") as f:
                f.write(labels_data)

        print("File generated successfully.")
        
        images_file_name = f"{self.images_file_prefix}-idx3-ubyte"
        if self.use_compression: images_file_name += ".gz"
        images_magic = 2051

        print(f"Generating file {images_file_name} with {size} images...")

        images_data = b''
        images_data += struct.pack(">I", images_magic)
        images_data += struct.pack(">I", size)
        images_data += struct.pack(">I", self.height)
        images_data += struct.pack(">I", self.width)
        for label in self.labels:
            images_data += b''.join(image.tobytes() for image in self.dataset[label])

        print("Writing data to file...")

        if self.use_compression: 
            with gzip.open(images_file_name, 'wb', compresslevel=6) as f:
                f.write(images_data)
        else:
            with open(images_file_name, "wb") as f:
                f.write(images_data)

        print("File generated successfully.")

        return

    def load_mnist_format_dataset(self):
        label_path = f"{self.labels_file_prefix}-idx1-ubyte"
        image_path = f"{self.images_file_prefix}-idx3-ubyte"
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
        print(f"magic: {magic}, size: {size}")
        labels = np.frombuffer(labels_data[8:], dtype=np.uint8).tolist()

        images = []
        images_data = b""
        with open(image_path, 'rb') as f:
            images_data = f.read()

        if self.use_compression: images_data = gzip.decompress(images_data)
        magic, size, rows, cols = struct.unpack('>IIII', images_data[:16])
        print(f"magic: {magic}, size: {size}, rows: {rows}, cols: {cols}")
        assert (magic == 2051)
        num_pixels = size * rows * cols
        image_data = np.frombuffer(images_data[16:16+num_pixels], dtype=np.uint8)
        images = image_data.reshape((size, rows, cols))
            
        return images, labels

    def show_dataset(self, x_train, y_train):
        mapping = ['g', 'y', 'b', 't', "invalid"]
        for idx, image_data in enumerate(x_train):
            # Display the image
            plt.imshow(image_data, cmap="gray", interpolation="nearest")
            plt.axis("off")  # Turn off axis
            plt.title(f"Image {idx} with label {y_train[idx]} - {mapping[y_train[idx]]}")
            plt.show()
        return

def find_dataset_images(folder_path, exclude_substring = ""):
    paths_and_labels = {}

    paths_and_labels[0] = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) 
        if f.endswith('.png') and f.startswith('g') and exclude_substring not in f
    ]
    
    paths_and_labels[1] = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) 
        if f.endswith('.png') and f.startswith('y') and exclude_substring not in f
    ]
    
    paths_and_labels[2] = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) 
        if f.endswith('.png') and f.startswith('b') and exclude_substring not in f
    ]

    paths_and_labels[3] = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) 
        if f.endswith('.png') and f.startswith('t') and exclude_substring not in f
    ]

    return paths_and_labels

if __name__ == "__main__":
    # Find all the paths except the ones used for the test dataset
    paths_and_labels = find_dataset_images("./example", "bpen")

    # First generate the dataset:              -- labels_prefix --                   -- images prefix --                             -- width & height --
    dataset_generator = DatasetGenerator("./dataset/my-dataset-train-labels", "./dataset/my-dataset-train-images", paths_and_labels,        28, 28, use_compression=True)
    # Generate multiple transformations of the given images, effectively populating the dataset (which at this point is a dictionary)
    dataset_generator.generate_dataset()
    # Save the dataset using the mnist format
    dataset_generator.store_dataset_as_mnist_format()


    # Do the same for the test dataset
    dataset_generator = DatasetGenerator("./dataset/my-dataset-test-labels", "./dataset/my-dataset-test-images", {0: ["./example/g_bpen.png"], 1: ["./example/y_bpen.png"], 2: ["./example/b_bpen.png"], 3: ["./example/t_bpen.png"]}, 28, 28, 2,use_compression=True)
    dataset_generator.generate_dataset()
    dataset_generator.store_dataset_as_mnist_format()

    # To load the dataset specify the prefixes as above
    dataset_viewer = DatasetGenerator("./dataset/my-dataset-train-labels", "./dataset/my-dataset-train-images",use_compression=True)
    # load the labels and the images
    x_train, y_train = dataset_generator.load_mnist_format_dataset()
    # Plot every image inside x_train
    dataset_generator.show_dataset(x_train, y_train)
