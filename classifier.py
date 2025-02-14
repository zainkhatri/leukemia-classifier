import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from datetime import datetime

class LeukemiaClassifier:
    def __init__(self, img_size=(224, 224), num_classes=4, backbone='resnet50', 
                 model_dir='models'):
        """
        Initialize the Leukemia Classifier
        
        Args:
            img_size (tuple): Input image dimensions (height, width)
            num_classes (int): Number of classification categories
            backbone (str): Neural network backbone ('resnet50', 'efficientnet')
            model_dir (str): Directory to save model checkpoints
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.backbone = backbone
        self.model_dir = model_dir
        self.class_weights = None
        self.history = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Build the model
        self.model = self.build_model()
        
    def build_model(self):
        """
        Build a simple CNN architecture with ResNet50 backbone
        
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        # Base model with pre-trained weights
        base_model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=self.img_size + (3,)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create a sequential model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model with only accuracy as a metric
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def compute_class_weights(self, y_train):
        """
        Compute class weights to handle class imbalance
        
        Args:
            y_train (np.array): Training labels
            
        Returns:
            dict: Class weights dictionary
        """
        # Count samples in each class
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        
        # Compute class weights inversely proportional to class frequency
        n_classes = len(class_counts)
        weights = {}
        
        for i in range(n_classes):
            # Skip classes with no samples
            if class_counts[i] > 0:
                # More weight to minority classes, less to majority classes
                weights[i] = (total_samples / (n_classes * class_counts[i]))
        
        self.class_weights = weights
        return weights
    
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=32, epochs=50, use_class_weights=True):
        """
        Train the model with data augmentation and callbacks
        
        Args:
            X_train (np.array): Training image data
            y_train (np.array): Training labels
            X_val (np.array): Validation image data
            y_val (np.array): Validation labels
            batch_size (int): Training batch size
            epochs (int): Maximum training epochs
            use_class_weights (bool): Whether to use class weights to handle imbalance
        
        Returns:
            dict: Training history
        """
        # Create a timestamp for unique model naming
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Compute class weights if requested
        if use_class_weights:
            class_weights = self.compute_class_weights(y_train)
            print("Using class weights:", class_weights)
        else:
            class_weights = None
        
        # Callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateauing
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2, 
                patience=5, 
                min_lr=1e-6,
                verbose=1
            ),
            
            # Save best model checkpoints
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f'leukemia_classifier_{timestamp}_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        print("\n---- Training model ----")
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=len(X_train) // batch_size,
            class_weight=class_weights,
            verbose=1
        )
        
        self.history = history.history
        return history.history
    
    def predict(self, X):
        """
        Perform prediction
        
        Args:
            X (np.array): Input images
        
        Returns:
            np.array: Predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, class_names):
        """
        Evaluate the model
        
        Args:
            X_test (np.array): Test image data
            y_test (np.array): Test labels
            class_names (list): List of class names for visualization
            
        Returns:
            tuple: (metrics, predictions)
        """
        # Get predictions
        predictions = self.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Basic metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Calculate confusion matrix manually
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        for i in range(len(y_test)):
            cm[y_test[i], y_pred[i]] += 1
            
        # Create a confusion matrix visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Add labels
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add values to the confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'))
        plt.show()
        
        # Calculate class-wise metrics
        class_metrics = {}
        for i in range(self.num_classes):
            # True positives, false positives, false negatives
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            # Calculate precision, recall, f1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_names[i]] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Print classification report
        print("\nClassification Report:")
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        for class_name, metrics in class_metrics.items():
            print(f"{class_name:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
        
        # Return evaluation metrics
        metrics = {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'class_metrics': class_metrics,
            'confusion_matrix': cm
        }
        
        return metrics, predictions
    
    def plot_training_history(self):
        """Plot training metrics from model history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        axes[0].plot(self.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(loc='lower right')
        axes[0].grid(True)
        
        # Plot training & validation loss
        axes[1].plot(self.history['loss'], label='Train Loss')
        axes[1].plot(self.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(loc='upper right')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        plt.show()
    
    def save_model(self, custom_name=None):
        """
        Save the trained model to disk
        
        Args:
            custom_name (str, optional): Custom filename for the model
            
        Returns:
            str: Path to the saved model
        """
        if custom_name:
            model_path = os.path.join(self.model_dir, f"{custom_name}.h5")
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_path = os.path.join(self.model_dir, f"leukemia_classifier_{timestamp}.h5")
            
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    @classmethod
    def load_model(cls, model_path, custom_config=None):
        """
        Load a previously saved model
        
        Args:
            model_path (str): Path to the saved model file
            custom_config (dict, optional): Custom configuration parameters
            
        Returns:
            LeukemiaClassifier: Initialized classifier with loaded model
        """
        # Create a default instance
        if custom_config is None:
            custom_config = {}
            
        # Create instance with configs
        img_size = custom_config.get('img_size', (224, 224))
        num_classes = custom_config.get('num_classes', 4)
        backbone = custom_config.get('backbone', 'resnet50')
        model_dir = custom_config.get('model_dir', 'models')
        
        # Create instance
        instance = cls(
            img_size=img_size,
            num_classes=num_classes,
            backbone=backbone,
            model_dir=model_dir
        )
        
        # Load the model
        instance.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        return instance


def main():
    """
    Example of using the LeukemiaClassifier
    """
    from data_preprocessing import LeukemiaDataProcessor
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate the Leukemia Classifier')
    parser.add_argument('--data_dir', type=str, default='Blood cell Cancer [ALL]',
                        help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (both width and height)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    print(f"Loading data from {args.data_dir}...")
    processor = LeukemiaDataProcessor(args.data_dir)
    
    # Load images and labels
    images, labels, class_names = processor.load_images(target_size=(args.img_size, args.img_size))
    print(f"Loaded {len(images)} images from {len(set(class_names))} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(images, labels)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Initialize model
    classifier = LeukemiaClassifier(
        img_size=(args.img_size, args.img_size),
        num_classes=len(set(labels)),
        model_dir=args.model_dir
    )
    
    # Train model
    print("\n---- Starting Training ----")
    history = classifier.train(
        X_train, y_train, X_test, y_test,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate model
    print("\n---- Evaluation ----")
    metrics, predictions = classifier.evaluate(
        X_test, y_test, list(processor.class_mapping.keys())
    )
    
    # Print key metrics
    print(f"\nTest Accuracy: {metrics['accuracy'] * 100:.2f}%")
    
    # Save the model
    model_path = classifier.save_model()
    
    print("\n---- Completed Successfully ----")
    print(f"Model saved to {model_path}")
    
    return classifier, metrics


if __name__ == '__main__':
    main()