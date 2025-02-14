import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LeukemiaDataProcessor:
    def __init__(self, data_dir):
        """
        Initialize the data processor with the directory containing image classes
        
        Args:
            data_dir (str): Path to the directory containing class subdirectories
        """
        import os
        
        self.data_dir = data_dir
        
        # Predefined classes to match the project specification
        self.classes = [
            'Benign', 
            '[Malignant] Pre-B', 
            '[Malignant] Pro-B', 
            '[Malignant] early Pre-B'
        ]
        
        # Verify all directories exist
        self.class_dirs = []
        for cls in self.classes:
            class_path = os.path.join(data_dir, cls)
            if os.path.exists(class_path):
                self.class_dirs.append(class_path)
                print(f"Found directory: {class_path}")
            else:
                print(f"WARNING: Directory not found: {class_path}")
        
        # Create a mapping to simplify class names for model training
        self.class_mapping = {
            'Benign': 0,
            '[Malignant] Pre-B': 1,
            '[Malignant] Pro-B': 2,
            '[Malignant] early Pre-B': 3
        }
        
    def load_images(self, target_size=(224, 224)):
        """
        Load images from directories and prepare dataset
        
        Args:
            target_size (tuple): Desired image size for resizing
        
        Returns:
            tuple: (images, labels, class_names)
        """
        images = []
        labels = []
        class_names = []
        
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found!")
                continue
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                
                try:
                    # Read and preprocess image
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, 
                        target_size=target_size
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0  # Normalize pixel values
                    
                    images.append(img_array)
                    # Use the mapped class index
                    labels.append(self.class_mapping[class_name])
                    class_names.append(class_name)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return (
            np.array(images), 
            np.array(labels), 
            class_names
        )
    
    def split_data(self, images, labels, test_size=0.2, random_state=42):
        """
        Split data into train and test sets while maintaining stratification
        
        Args:
            images (np.array): Image data
            labels (np.array): Corresponding labels
            test_size (float): Proportion of the dataset to include in test split
            random_state (int): Controls the shuffling applied to the data
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(
            images, 
            labels, 
            test_size=test_size, 
            stratify=labels, 
            random_state=random_state
        )
    
    def create_data_augmentation(self):
        """
        Create data augmentation generator
        
        Returns:
            ImageDataGenerator: Configured data augmentation generator
        """
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    def visualize_class_distribution(self, labels, class_names):
        """
        Create a bar plot showing class distribution
        
        Args:
            labels (np.array): Numeric labels
            class_names (list): Corresponding class names
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        # Sort the unique labels to ensure consistent order
        sorted_indices = np.argsort(unique)
        unique = unique[sorted_indices]
        counts = counts[sorted_indices]
        
        plt.figure(figsize=(10, 5))
        plt.bar(
            [class_names[unique[i]] for i in range(len(unique))], 
            counts
        )
        plt.title('Class Distribution in Leukemia Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    data_dir = 'Blood cell Cancer [ALL]'
    print("Checking directories in:", data_dir)
    print("Available directories:")
    for item in os.listdir(data_dir):
        print(item)
    
    processor = LeukemiaDataProcessor(data_dir)
    
    # Load images
    images, labels, class_names = processor.load_images()
    
    # Visualize class distribution
    processor.visualize_class_distribution(labels, list(set(class_names)))
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(images, labels)
    
    # Create data augmentation generator
    aug_generator = processor.create_data_augmentation()
    
    # Detailed class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nDetailed Class Distribution:")
    for i in range(len(unique)):
        print(f"{list(set(class_names))[unique[i]]}: {counts[i]} images")
    
    print(f"\nTotal images: {len(images)}")
    print(f"Training images: {len(X_train)}")
    print(f"Test images: {len(X_test)}")
    
    # Verification of class balance in train and test sets
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    
    print("\nTraining Set Class Distribution:")
    for i in range(len(train_unique)):
        print(f"{list(set(class_names))[train_unique[i]]}: {train_counts[i]} images")
    
    print("\nTest Set Class Distribution:")
    for i in range(len(test_unique)):
        print(f"{list(set(class_names))[test_unique[i]]}: {test_counts[i]} images")

if __name__ == '__main__':
    main()