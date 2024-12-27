import cv2
import numpy as np
import random
import struct
import matplotlib.pyplot as plt

class Transformations:
    def __init__(self, images): 
        self.images = images
        pass

    def apply_scaling(self, scales=[0.8, 1.0, 1.2]):
        for scale in scales:
            for img in self.images[:]:
                scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                self.images.append(scaled)

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

    def apply_contrast_adjustment(self, contrast_factors=[0.8, 1.0, 1.2]):
        for alpha in contrast_factors:
            for img in self.images[:]:
                adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
                self.images.append(adjusted)

    def apply_gaussian_noise(self, noise_levels=[10, 20]):
        for noise_level in noise_levels:
            for img in self.images[:]:
                noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
                noisy_image = cv2.add(img, noise)
                self.images.append(noisy_image)

    def apply_all_transformations(self):
        print("applying scaling...")
        self.apply_scaling()
        # print("applying translation...")
        # self.apply_translation()
        # print("applying shearing...")
        # self.apply_shearing()
        print("applying rotation...")
        self.apply_rotation()
        print("applying flipping...")
        self.apply_flipping()
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
    def __init__(self, labels_file_prefix, images_file_prefix, images_paths = [], labels = [], width = 28, height = 28, noise_cnt = 10):
        self.labels_file_prefix = labels_file_prefix
        assert (self.labels_file_prefix != "")
        self.images_file_prefix = images_file_prefix
        assert (self.images_file_prefix != "")

        self.dataset = {}
        self.width = width
        self.height = height
        self.labels = labels

        self.images = []
        self.images_paths = images_paths
        if self.images_paths != []: 
            self.read_images()
        
        self.noise_images = []
        for _ in range(noise_cnt):
            self.noise_images.append(self.generate_random_noise())

        pass
    
    def read_images(self):
        for image_path in self.images_paths:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (self.width, self.height))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            self.images.append(binary)
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
        for idx, img in enumerate(self.images):
            print(f"Generating images from transformation of image {idx}...")
            img_transformations = Transformations([img])
            img_transformations.apply_all_transformations()
            self.dataset[self.labels[idx]] = img_transformations.get_images()
            print(f"Successfully generated {len(self.dataset[self.labels[idx]])} images.")
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
        labels_magic = 2049
        
        size = 0
        for label in self.labels:
            size += len(self.dataset[label])

        print(f"Generating file {labels_file_name} with {size} labels...")
        with open(labels_file_name, "wb") as file:
            file.write(struct.pack(">I", labels_magic))
            file.write(struct.pack(">I", size))
            for label in self.labels:
                for _ in range(len(self.dataset[label])):
                    file.write(struct.pack(">B", label))

        print("File generated successfully.")
        
        images_file_name = f"{self.images_file_prefix}-idx3-ubyte"
        images_magic = 2051

        print(f"Generating file {images_file_name} with {size} images...")

        with open(images_file_name, "wb") as file:
            file.write(struct.pack(">I", images_magic))
            file.write(struct.pack(">I", size))
            file.write(struct.pack(">I", self.height))
            file.write(struct.pack(">I", self.width))
            for label in self.labels:
                for image in self.dataset[label]:
                    image = cv2.resize(image, (self.width, self.height))
                    file.write(image.tobytes())

        print("File generated successfully.")

        return

    def load_mnist_format_dataset(self):
        label_path = f"{self.labels_file_prefix}-idx1-ubyte"
        image_path = f"{self.images_file_prefix}-idx3-ubyte"

        labels = []
        with open(label_path, 'rb') as f:
            magic, size = struct.unpack('>II', f.read(8))
            assert (magic == 2049)
            print(f"magic: {magic}, size: {size}")
            labels = np.frombuffer(f.read(size), dtype=np.uint8).tolist()

        images = []
        with open(image_path, 'rb') as f:
            # Read the header: magic number, size, rows, cols
            magic, size, rows, cols = struct.unpack('>IIII', f.read(16))
            
            # Validate the magic number
            if magic != 2051:
                raise ValueError(f"Invalid magic number: {magic}, expected 2051")
            
            # Read the rest of the file as image data
            num_pixels = size * rows * cols
            image_data = np.frombuffer(f.read(num_pixels), dtype=np.uint8)
            
            # Reshape into (size, rows, cols)
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

if __name__ == "__main__":
    # First generate the dataset:              -- labels_prefix --                   -- images prefix --               -- path of the letters classes --                                                                     -- number of each letter class --   -- width & height --  
    dataset_generator = DatasetGenerator("./dataset/my-dataset-train-labels", "./dataset/my-dataset-train-images", ["./dataset/g_pencil.png", "./dataset/y_pencil.png", "./dataset/b_pencil.png", "./dataset/t_pencil.png"],             [0, 1, 2, 3],                  28, 28)
    # Generate multiple transformations of the given images, effectively populating the dataset (which at this point is a dictionary)
    dataset_generator.generate_dataset()
    # Save the dataset using the mnist format
    dataset_generator.store_dataset_as_mnist_format()

    # To load the dataset specify the prefixes as above
    dataset_viewer = DatasetGenerator("./dataset/my-dataset-train-labels", "./dataset/my-dataset-train-images")
    # load the labels and the images
    x_train, y_train = dataset_generator.load_mnist_format_dataset()
    # Plot every image inside x_train
    dataset_generator.show_dataset(x_train, y_train)
