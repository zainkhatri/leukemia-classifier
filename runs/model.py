import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input, regularizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,
    Conv2D, Activation, Multiply, Reshape, Lambda, AveragePooling2D,
    MaxPooling2D, Concatenate, Add, UpSampling2D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    TensorBoard, LearningRateScheduler
)
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from datetime import datetime
import time
import logging
import json

# Import custom layers
from runs.layers import AttentionModule, FocalLoss

# Check for TFA
try:
    import tensorflow_addons as tfa
    from tensorflow_addons.layers import SpectralNormalization
    from tensorflow_addons.optimizers import RectifiedAdam
    HAVE_TFA = True
except ImportError:
    print("WARNING: TensorFlow Addons not available. Using standard optimizers instead.")
    HAVE_TFA = False

class LeukemiaClassifierV5:
    # Enhanced leukemia classifier
    def __init__(self, img_size=(224, 224), num_classes=2, 
                 backbone='efficientnet', model_dir='models/v5', 
                 binary=True, learning_rate=1e-4, dropout_rate=0.5,
                 l2_reg=1e-5, use_attention=True, use_focal_loss=True):
        # Init model params
        self.img_size = img_size
        self.num_classes = 2 if binary else num_classes
        self.binary = binary
        self.backbone = backbone
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        self.use_focal_loss = use_focal_loss
        
        # Other init vars
        self.class_weights = None
        self.history = None
        self.models = []  # For ensemble
        self.mc_samples = 30  # MC dropout samples
        self.decision_threshold = 0.5  # Binary threshold
        
        # Create directories
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Build model
        self.model = self.build_model()
        
        # Log config
        self.logger.info(f"Created {self.__class__.__name__} with {backbone} backbone")
        self.logger.info(f"Input size: {img_size}, Classes: {num_classes}, Binary: {binary}")
        self.logger.info(f"Using attention: {use_attention}, Using focal loss: {use_focal_loss}")
    
    def _create_directories(self):
        # Create model dirs
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'predictions'), exist_ok=True)
    
    def _setup_logging(self):
        # Configure logger
        log_file = os.path.join(self.model_dir, 'logs', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # Create logger
        self.logger = logging.getLogger('LeukemiaClassifierV5')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def build_model(self):
        # Create model architecture
        if self.binary:
            return self.build_binary_model()
        else:
            return self.build_multiclass_model()
    
    def build_binary_model(self):
        # Build binary classifier
        # Input layer
        inputs = Input(shape=self.img_size + (3,), name="input_layer")
        
        # Get backbone
        if self.backbone == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs,
                drop_connect_rate=0.2  # Path dropout
            )
        elif self.backbone == 'resnet':
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Store backbone
        self.base_model = base_model
        
        # Freeze backbone
        base_model.trainable = False
        
        # Get backbone output
        x = base_model.output
        
        # Add attention
        if self.use_attention:
            x = AttentionModule(reduction_ratio=8, kernel_size=7)(x)
        
        # Global pooling
        x = GlobalAveragePooling2D()(x)
        
        # Normalize features
        x = BatchNormalization()(x)
        
        # First dense layer
        x = Dense(
            256, 
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="features_dense_1"
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Second dense layer
        x = Dense(
            128, 
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="features_dense_2"
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        if self.binary:
            outputs = Dense(
                1, 
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name="output_layer"
            )(x)
        else:
            outputs = Dense(
                self.num_classes, 
                activation='softmax',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name="output_layer"
            )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="LeukemiaClassifierV5")
        
        # Compile model
        self._compile_model(model)
        
        return model
    
    def build_multiclass_model(self):
        # Build multi-class model
        # Input layer
        inputs = Input(shape=self.img_size + (3,), name="input_layer")
        
        # Get backbone
        if self.backbone == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs,
                drop_connect_rate=0.2
            )
        elif self.backbone == 'resnet':
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_tensor=inputs
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Store backbone
        self.base_model = base_model
        
        # Freeze backbone
        base_model.trainable = False
        
        # Get backbone output
        x = base_model.output
        
        # Add attention
        if self.use_attention:
            x = AttentionModule(reduction_ratio=8, kernel_size=7)(x)
        
        # Global pooling
        x = GlobalAveragePooling2D()(x)
        
        # Normalize features
        x = BatchNormalization()(x)
        
        # First dense layer
        x = Dense(
            256, 
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="features_dense_1"
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Second dense layer
        x = Dense(
            128, 
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="features_dense_2"
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output for multi-class
        outputs = Dense(
            self.num_classes, 
            activation='softmax',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="output_layer"
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name="LeukemiaMultiClassifierV5")
        
        # Compile model
        self._compile_model(model)
        
        return model
    
    def _compile_model(self, model):
        # Setup optimizer/loss
        # Select optimizer
        if HAVE_TFA:
            optimizer = tfa.optimizers.RectifiedAdam(
                learning_rate=self.learning_rate,
                weight_decay=self.l2_reg,
                clipnorm=1.0,
                clipvalue=0.5
            )
        else:
            optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0,
                clipvalue=0.5
            )
        
        # Select loss function
        if self.use_focal_loss:
            # Use focal loss
            if self.binary:
                loss = FocalLoss(alpha=0.25, gamma=2.0)
            else:
                loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            # Use standard loss
            if self.binary:
                loss = 'binary_crossentropy'
            else:
                loss = 'categorical_crossentropy'
        
        # Setup metrics
        metrics = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
        
        # Add F1 score
        if HAVE_TFA:
            metrics.append(tfa.metrics.F1Score(
                num_classes=self.num_classes,
                threshold=0.5,
                average='macro',
                name='f1'
            ))
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def get_callbacks(self, monitor='val_loss', patience_early=20, patience_lr=10, 
                     fold_num=None, use_cosine_decay=True):
        # Get training callbacks
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fold_str = f"_fold{fold_num}" if fold_num is not None else ""
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor=monitor,
                patience=patience_early,
                restore_best_weights=True,
                verbose=1,
                mode='min' if 'loss' in monitor else 'max'
            ),
            
            # NaN detection
            tf.keras.callbacks.TerminateOnNaN(),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=os.path.join(
                    self.model_dir, 
                    'checkpoints', 
                    f'leukemia_v5{fold_str}_{timestamp}_best.h5'
                ),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='min' if 'loss' in monitor else 'max'
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(self.model_dir, 'logs', f'run{fold_str}_{timestamp}'),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        # Add LR scheduler
        if use_cosine_decay:
            # Cosine decay
            callbacks.append(
                LearningRateScheduler(
                    lambda epoch, lr: self._cosine_decay_with_warmup(epoch, lr, 
                                                                   total_epochs=100,
                                                                   warmup_epochs=5)
                )
            )
        else:
            # Reduce on plateau
            callbacks.append(
                ReduceLROnPlateau(
                    monitor=monitor,
                    factor=0.5,
                    patience=patience_lr,
                    min_lr=1e-7,
                    verbose=1,
                    mode='min' if 'loss' in monitor else 'max'
                )
            )
        
        return callbacks
    
    def _cosine_decay_with_warmup(self, epoch, lr, total_epochs=100, warmup_epochs=5):
        # LR schedule
        # Warmup phase
        if epoch < warmup_epochs:
            return self.learning_rate * ((epoch + 1) / warmup_epochs)
        
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return self.learning_rate * (0.5 * (1 + np.cos(np.pi * progress)))
    
    def compute_class_weights(self, y_train, beta=0.999):
        # Calculate class weights
        # Get label format
        y_train_int = y_train
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train_int = np.argmax(y_train, axis=1)
            
        # Count per class
        class_counts = np.bincount(y_train_int, minlength=self.num_classes)
        
        # Handle empty classes
        for i in range(len(class_counts)):
            if class_counts[i] == 0:
                class_counts[i] = 1  # Avoid division by zero
        
        # Effective sample numbers
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        
        # Normalize weights
        weights = weights / np.sum(weights) * len(class_counts)
        
        # Cap minority weight
        if self.binary and len(class_counts) == 2:
            # Find minority
            minority_idx = np.argmin(class_counts)
            
            # Calculate imbalance
            imbalance_ratio = class_counts[1 - minority_idx] / class_counts[minority_idx]
            
            # Cap minority weight
            max_weight = min(5.0, imbalance_ratio)
            weights[minority_idx] = min(weights[minority_idx], max_weight)
                
            # Renormalize
            weights = weights / np.sum(weights) * len(class_counts)
        
        # Create weight dict
        weight_dict = {i: float(w) for i, w in enumerate(weights)}
        
        self.logger.info(f"Class counts: {class_counts}")
        self.logger.info(f"Class weights: {weight_dict}")
        
        self.class_weights = weight_dict
        return weight_dict
    
    def train_single_model(self, X_train, X_val, y_train, y_val, 
                          batch_size=32, epochs=50, use_class_weights=True, 
                          use_augmentation=True, use_mixup=True, fine_tuning=True):
        # Train single model
        # Start timer
        start_time = time.time()
        
        # Format labels
        y_train, y_val = self._prepare_labels(y_train, y_val)
        
        # Get class weights
        class_weights = None
        if use_class_weights:
            class_weights = self.compute_class_weights(y_train)
        
        # Setup data augmentation
        if use_augmentation:
            # With augmentation
            datagen = self._get_data_generator(use_mixup=use_mixup)
            
            # Setup train gen
            train_generator = datagen.flow(
                X_train, y_train,
                batch_size=batch_size,
                shuffle=True
            )
            
            # Set steps
            steps_per_epoch = len(X_train) // batch_size
        else:
            # No augmentation
            train_generator = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
                .batch(batch_size) \
                .prefetch(tf.data.AUTOTUNE)
            steps_per_epoch = len(X_train) // batch_size
        
        # Get callbacks
        callbacks = self.get_callbacks(use_cosine_decay=True)
        
        # Phase 1: Frozen backbone
        self.logger.info("Phase 1: Training with frozen backbone")
        
        # Ensure frozen
        self.base_model.trainable = False
        
        try:
            # Initial training
            initial_epochs = 20 if fine_tuning else epochs
            
            history1 = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=initial_epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Save initial
            initial_model_path = os.path.join(
                self.model_dir, 'checkpoints', 'initial_phase_model.h5'
            )
            self.model.save(initial_model_path)
            self.logger.info(f"Initial phase model saved to {initial_model_path}")
            
        except Exception as e:
            self.logger.error(f"Error during initial training phase: {e}")
            self.logger.error("Initial training failed. Saving model in current state.")
            self.model.save(os.path.join(self.model_dir, 'checkpoints', 'failed_initial_model.h5'))
            
            # Return failed results
            elapsed_time = time.time() - start_time
            self.logger.info(f"Training failed after {elapsed_time:.2f} seconds")
            
            return (
                {}, 
                {
                    'accuracy': 0.0, 
                    'precision': 0.0, 
                    'recall': 0.0, 
                    'f1': 0.0,
                    'status': 'failed_initial'
                }
            )
        
        # Phase 2: Fine-tuning
        if fine_tuning:
            self.logger.info("Phase 2: Fine-tuning with gradual unfreezing")
            
            try:
                # Unfreeze layers
                self._gradual_unfreeze()
                
                # Reduce learning rate
                current_lr = float(self.model.optimizer.learning_rate)
                self.model.optimizer.learning_rate.assign(current_lr * 0.1)
                self.logger.info(f"Reduced learning rate to {current_lr * 0.1}")
                
                # Continue training
                history2 = self.model.fit(
                    train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    initial_epoch=history1.epoch[-1] + 1,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=1
                )
                
                # Combine histories
                combined_history = {}
                for key in history1.history:
                    combined_history[key] = history1.history[key] + history2.history[key]
                
                self.history = combined_history
                
            except Exception as e:
                self.logger.error(f"Error during fine-tuning phase: {e}")
                self.logger.error("Fine-tuning failed. Using model from initial phase.")
                
                # Try to recover
                try:
                    self.model = tf.keras.models.load_model(
                        initial_model_path, 
                        custom_objects={'FocalLoss': FocalLoss, 'AttentionModule': AttentionModule}
                    )
                except:
                    self.logger.error("Could not reload initial model.")
                
                self.history = history1.history
        else:
            self.history = history1.history
        
        # Log time
        elapsed_time = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        
        # Plot history
        self.plot_training_history()
        
        # Save model
        final_model_path = os.path.join(self.model_dir, 'checkpoints', 'final_model.h5')
        self.model.save(final_model_path)
        self.logger.info(f"Final model saved to {final_model_path}")
        
        # Evaluate model
        self.logger.info("Evaluating model on validation data")
        metrics = self.evaluate(X_val, y_val)
        
        # Optimize threshold
        if self.binary:
            self.logger.info("Optimizing decision threshold")
            self.optimize_threshold(X_val, y_val)
        
        # Add metadata
        metrics['training_time'] = elapsed_time
        metrics['status'] = 'completed'
        
        return self.history, metrics
    
    def _prepare_labels(self, y_train, y_val):
        # Format labels
        # Binary format
        if self.binary:
            # Convert one-hot
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                y_train = np.argmax(y_train, axis=1)
            if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                y_val = np.argmax(y_val, axis=1)
            
            # Flatten if needed
            if len(y_train.shape) > 1:
                y_train = y_train.flatten()
            if len(y_val.shape) > 1:
                y_val = y_val.flatten()
        else:
            # Multi-class format
            if len(y_train.shape) == 1 or y_train.shape[1] == 1:
                y_train = to_categorical(y_train, self.num_classes)
            if len(y_val.shape) == 1 or y_val.shape[1] == 1:
                y_val = to_categorical(y_val, self.num_classes)
        
        return y_train, y_val
    
    def _get_data_generator(self, use_mixup=True):
        # Create data augmenter
        # Import generator
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Cell image augmentation
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.15,
            height_shift_range=0.15,
            brightness_range=[0.8, 1.2],
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect',
            preprocessing_function=self._augmentation_function if use_mixup else None
        )
        
        return datagen
    
    def _augmentation_function(self, img):
        # Custom augmentation
        import cv2
        
        # Random augmentations
        
        # Color channel adjust
        if np.random.rand() < 0.3:
            # Adjust color balance
            r, g, b = cv2.split(img)
            
            # Random adjustments
            r = np.clip(r * np.random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
            g = np.clip(g * np.random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
            b = np.clip(b * np.random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
            
            img = cv2.merge([r, g, b])
        
        # Contrast adjustment
        if np.random.rand() < 0.3:
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=np.random.uniform(1.0, 3.0), 
                                    tileGridSize=(8, 8))
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            # Get L channel
            L_channel = lab[..., 0].copy().astype(np.uint8)
            # Apply to L only
            lab[..., 0] = clahe.apply(L_channel)
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Random sharpening
        if np.random.rand() < 0.2:
            # Apply sharpening
            blurred = cv2.GaussianBlur(img, (0, 0), np.random.uniform(1.0, 3.0))
            img = cv2.addWeighted(
                img, 1 + np.random.uniform(0.1, 0.5), 
                blurred, -np.random.uniform(0.1, 0.5), 0
            )
        
        return img
    
    def _gradual_unfreeze(self):
        # Unfreeze backbone layers
        if not hasattr(self, 'base_model'):
            self.logger.warning("No base model available for unfreezing")
            return
        
        # Count layers
        base_model = self.base_model
        total_layers = len(base_model.layers)
        self.logger.info(f"Base model has {total_layers} layers")
        
        # Get layers to unfreeze
        num_to_unfreeze = max(1, int(total_layers * 0.3))
        
        # Enable training
        base_model.trainable = True
        
        # Freeze all first
        for layer in base_model.layers:
            layer.trainable = False
        
        # Unfreeze last layers
        for layer in base_model.layers[-num_to_unfreeze:]:
            layer.trainable = True
            self.logger.debug(f"Unfreezing layer: {layer.name}")
        
        self.logger.info(f"Unfroze last {num_to_unfreeze} layers of the base model")
        
        # Recompile model
        self._compile_model(self.model)
    
    def train_ensemble(self, X_train, X_val, y_train, y_val, n_models=3, 
                      batch_size=32, epochs=50, use_augmentation=True):
        # Train model ensemble
        self.logger.info(f"Training ensemble of {n_models} models")
        
        # Save original
        original_model = self.model
        
        # Train models
        self.models = []
        ensemble_metrics = []
        
        for i in range(n_models):
            self.logger.info(f"Training ensemble model {i+1}/{n_models}")
            
            # Create new model
            self.model = self.build_model()
            
            # Set different seed
            seed = 42 + i
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            # Train model
            history, metrics = self.train_single_model(
                X_train, X_val, y_train, y_val,
                batch_size=batch_size,
                epochs=epochs,
                use_augmentation=use_augmentation,
                fine_tuning=True
            )
            
            # Save model
            model_path = os.path.join(self.model_dir, 'checkpoints', f'ensemble_model_{i+1}.h5')
            self.model.save(model_path)
            
            # Add to ensemble
            self.models.append(self.model)
            ensemble_metrics.append(metrics)
            
            self.logger.info(f"Ensemble model {i+1} validation accuracy: {metrics['accuracy']:.4f}")
        
        # Restore original
        self.model = original_model
        
        # Evaluate ensemble
        self.logger.info("Evaluating ensemble performance")
        ensemble_metrics = self.evaluate_ensemble(X_val, y_val)
        
        self.logger.info(f"Ensemble validation accuracy: {ensemble_metrics['accuracy']:.4f}")
        self.logger.info(f"Ensemble validation AUC: {ensemble_metrics['auc']:.4f}")
        
        return self.models, ensemble_metrics
    
    def train_with_kfold(self, images, labels, n_folds=5, batch_size=32, epochs=50,
                        use_class_weights=True, use_augmentation=True, fine_tuning=True):
        # Train with cross-validation
        from sklearn.model_selection import StratifiedKFold
        
        self.logger.info(f"Starting {n_folds}-fold cross-validation")
        
        # Prepare labels
        if self.binary:
            # Get 1D labels
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                y = np.argmax(labels, axis=1)
            else:
                y = labels.flatten() if len(labels.shape) > 1 else labels
        else:
            # Get class indices
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                y = np.argmax(labels, axis=1)
            else:
                y = labels
        
        # Init k-fold
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Store results
        fold_metrics = []
        self.models = []
        fold_histories = []
        
        # Save original
        original_model = self.model
        
        # Train each fold
        for fold, (train_idx, val_idx) in enumerate(kfold.split(images, y)):
            self.logger.info(f"Training fold {fold+1}/{n_folds}")
            
            # Split for fold
            X_train, X_val = images[train_idx], images[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Create new model
            self.model = self.build_model()
            
            # Train fold model
            history, metrics = self.train_single_model(
                X_train, X_val, y_train, y_val,
                batch_size=batch_size,
                epochs=epochs,
                use_class_weights=use_class_weights,
                use_augmentation=use_augmentation,
                fine_tuning=fine_tuning
            )
            
            # Save fold model
            model_path = os.path.join(self.model_dir, 'checkpoints', f'fold_{fold+1}_model.h5')
            self.model.save(model_path)
            
            # Add to ensemble
            self.models.append(self.model)
            fold_metrics.append(metrics)
            fold_histories.append(history)
            
            self.logger.info(f"Fold {fold+1} validation metrics: {metrics}")
        
        # Restore original
        self.model = original_model
        
        # Calculate CV metrics
        cv_results = self._calculate_cv_metrics(fold_metrics)
        self.logger.info(f"Cross-validation results: {cv_results}")
        
        # Get average threshold
        if self.binary:
            thresholds = [m.get('threshold', 0.5) for m in fold_metrics if 'threshold' in m]
            if thresholds:
                self.decision_threshold = np.mean(thresholds)
                self.logger.info(f"Average optimal threshold: {self.decision_threshold:.4f}")
            else:
                self.logger.warning("No optimal thresholds found in fold metrics")
        
        # Return CV results
        return cv_results, self.models
    
    def _calculate_cv_metrics(self, fold_metrics):
        # Average fold metrics
        # Init result dict
        cv_results = {}
        
        # Get metric keys
        metric_keys = set()
        for metrics in fold_metrics:
            metric_keys.update(metrics.keys())
        
        # Remove non-numeric
        non_numeric_keys = {'status', 'confusion_matrix'}
        metric_keys = metric_keys - non_numeric_keys
        
        # Calculate mean/std
        for key in metric_keys:
            # Get values
            values = [m[key] for m in fold_metrics if key in m]
            
            if values:
                cv_results[f'{key}_mean'] = np.mean(values)
                cv_results[f'{key}_std'] = np.std(values)
        
        return cv_results
    
    def optimize_threshold(self, X_val, y_val):
        # Find optimal threshold
        if not self.binary:
            self.logger.info("Threshold optimization is only for binary classification")
            return 0.5
        
        # Get 1D labels
        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            y_true = np.argmax(y_val, axis=1)
        else:
            y_true = y_val.flatten() if len(y_val.shape) > 1 else y_val
        
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        # Get predictions
        try:
            y_pred_proba = self.predict_proba(X_val)
            
            # Check for NaN
            if np.isnan(y_pred_proba).any():
                self.logger.warning("NaN values in predictions. Using default threshold of 0.5")
                return 0.5
            
            # Try different methods
            
            # F1 optimization
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
            
            # Calculate F1
            f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
            
            # Get best F1
            best_f1_idx = np.argmax(f1_scores)
            if best_f1_idx < len(thresholds_pr):
                best_f1_threshold = thresholds_pr[best_f1_idx]
            else:
                # Edge case
                best_f1_threshold = 0.5
            
            # Balanced accuracy
            fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
            
            # Calculate balanced acc
            balanced_acc = (tpr + (1 - fpr)) / 2
            
            # Get best BA
            best_ba_idx = np.argmax(balanced_acc)
            if best_ba_idx < len(thresholds_roc):
                best_ba_threshold = thresholds_roc[best_ba_idx]
            else:
                best_ba_threshold = 0.5
            
            # Custom scoring
            custom_scores = (0.7 * f1_scores[:len(thresholds_pr)] + 
                            0.3 * balanced_acc[:len(thresholds_pr)] 
                            if len(thresholds_pr) == len(balanced_acc) 
                            else f1_scores[:len(thresholds_pr)])
            
            best_custom_idx = np.argmax(custom_scores)
            if best_custom_idx < len(thresholds_pr):
                best_custom_threshold = thresholds_pr[best_custom_idx]
            else:
                best_custom_threshold = 0.5
            
            # Log thresholds
            self.logger.info(f"Best F1 threshold: {best_f1_threshold:.4f} (F1: {f1_scores[best_f1_idx]:.4f})")
            self.logger.info(f"Best balanced accuracy threshold: {best_ba_threshold:.4f} (BA: {balanced_acc[best_ba_idx]:.4f})")
            self.logger.info(f"Best custom threshold: {best_custom_threshold:.4f}")
            
            # Use F1 threshold
            best_threshold = best_f1_threshold
            
            # Set threshold
            self.decision_threshold = best_threshold
            self.logger.info(f"Using optimal threshold: {self.decision_threshold:.4f}")
            
            # Plot curves
            self._plot_threshold_optimization(y_true, y_pred_proba, best_threshold)
            
            return best_threshold
            
        except Exception as e:
            self.logger.error(f"Error optimizing threshold: {e}")
            self.logger.warning("Using default threshold of 0.5")
            return 0.5
    
    def _plot_threshold_optimization(self, y_true, y_pred_proba, best_threshold):
        # Plot threshold curves
        from sklearn.metrics import precision_recall_curve, roc_curve, auc
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        # Mark best threshold
        idx = np.abs(thresholds - best_threshold).argmin()
        ax1.plot(fpr[idx], tpr[idx], 'ro', label=f'Threshold = {best_threshold:.3f}')
        
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # PR Curve
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = auc(recall, precision)
        
        ax2.plot(recall, precision, color='green', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Mark threshold
        if len(thresholds_pr) > 0:
            # Handle length mismatch
            if len(thresholds_pr) < len(precision):
                thresholds_pr = np.append(thresholds_pr, thresholds_pr[-1])
            
            idx = np.abs(thresholds_pr - best_threshold).argmin()
            ax2.plot(recall[idx], precision[idx], 'ro', label=f'Threshold = {best_threshold:.3f}')
        
        # F1 isometrics
        f1_values = np.linspace(0.1, 0.9, num=9)
        for f1_value in f1_values:
            x = np.linspace(0.01, 1)
            y = (f1_value * x) / (2 * x - f1_value)
            mask = ~np.isnan(y)
            ax2.plot(x[mask], y[mask], color='gray', alpha=0.2)
            # Add small f1 label
            rightmost_idx = np.where(mask)[0][-1]
            ax2.annotate(f'f1={f1_value:.1f}', 
                        xy=(x[rightmost_idx], y[rightmost_idx]),
                        xytext=(x[rightmost_idx], y[rightmost_idx]),
                        fontsize=8, alpha=0.6)
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'visualizations', 'threshold_optimization.png'), dpi=300)
        plt.close()
    
    def predict(self, X, threshold=None):
        # Get class predictions
        # Use default threshold
        if threshold is None:
            threshold = self.decision_threshold
        
        # Make predictions
        if self.binary:
            # Binary prediction
            y_pred_proba = self.predict_proba(X)
            y_pred = (y_pred_proba > threshold).astype(int)
            return y_pred
        else:
            # Multi-class prediction
            y_pred_proba = self.predict_proba(X)
            y_pred = np.argmax(y_pred_proba, axis=1)
            return y_pred
    
    def predict_proba(self, X, use_tta=False, n_tta=5):
        # Get class probabilities
        if use_tta:
            return self.predict_with_tta(X, n_tta)
        
        # Use ensemble if available
        if len(self.models) > 0:
            return self.predict_with_ensemble(X)
        
        # Standard predict
        if self.binary:
            return self.model.predict(X).flatten()
        else:
            return self.model.predict(X)
    
    def predict_with_uncertainty(self, X, mc_samples=None):
        # MC dropout predictions
        if mc_samples is None:
            mc_samples = self.mc_samples
        
        # Collect samples
        y_pred_samples = []
        
        for _ in range(mc_samples):
            y_pred = self.model(X, training=True).numpy()  # Keep dropout
            y_pred_samples.append(y_pred)
        
        # Stack predictions
        y_pred_samples = np.stack(y_pred_samples, axis=0)
        
        # Binary case
        if self.binary:
            # Flatten binary dim
            y_pred_samples = y_pred_samples.squeeze(axis=-1)
            
            # Get stats
            y_pred_mean = np.mean(y_pred_samples, axis=0)
            y_pred_std = np.std(y_pred_samples, axis=0)
            
            # Apply threshold
            y_pred_binary = (y_pred_mean > self.decision_threshold).astype(int)
            
            return y_pred_mean, y_pred_binary, y_pred_std
        
        # Multi-class case
        else:
            # Get means
            y_pred_mean = np.mean(y_pred_samples, axis=0)
            
            # Entropy as uncertainty
            epsilon = 1e-10  # Avoid log(0)
            predictive_entropy = -np.sum(y_pred_mean * np.log(y_pred_mean + epsilon), axis=-1)
            
            # Get classes
            y_pred_class = np.argmax(y_pred_mean, axis=-1)
            
            return y_pred_mean, y_pred_class, predictive_entropy
    
    def predict_with_ensemble(self, X):
        # Ensemble predictions
        if not self.models:
            self.logger.warning("No ensemble models available. Using single model.")
            return self.model.predict(X)
        
        all_predictions = []
        
        # Get predictions
        for model in self.models:
            pred = model.predict(X)
            
            # Flatten binary
            if self.binary:
                pred = pred.flatten()
            
            all_predictions.append(pred)
        
        # Stack all
        all_predictions = np.array(all_predictions)
        
        # Get mean
        y_pred_mean = np.mean(all_predictions, axis=0)
        
        return y_pred_mean
    
    def predict_with_tta(self, X, n_augmentations=5):
        # Test-time augmentation
        # Import processor
        from runs.data_processing import LeukemiaDataProcessor
        
        # Create processor
        processor = LeukemiaDataProcessor(data_dir=".")  # Dummy dir
        
        # Get augmentations
        tta_images = processor.get_tta_batches(X, n_augmentations)
        
        all_predictions = []
        
        # Get predictions
        for aug_images in tta_images:
            pred = self.model.predict(aug_images)
            
            # Flatten binary
            if self.binary:
                pred = pred.flatten()
                
            all_predictions.append(pred)
        
        # Stack and average
        all_predictions = np.array(all_predictions)
        y_pred_mean = np.mean(all_predictions, axis=0)
        
        return y_pred_mean
    
    def evaluate(self, X_test, y_test, class_names=None, use_tta=False, 
                visualize=True, detailed=True):
        # Evaluate model
        from sklearn.metrics import (
            precision_recall_curve, roc_curve, auc, confusion_matrix, 
            classification_report, precision_score, recall_score, f1_score,
            average_precision_score, roc_auc_score, accuracy_score
        )
        
        # Default metrics
        default_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'f1': 0.0,
            'auc': 0.5,
            'threshold': self.decision_threshold,
            'confusion_matrix': None
        }
        
        # Format labels
        if self.binary:
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_true = np.argmax(y_test, axis=1)
            else:
                y_true = y_test.flatten() if len(y_test.shape) > 1 else y_test
        else:
            # Get class indices
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_true = np.argmax(y_test, axis=1)
            else:
                y_true = y_test
        
        # Make predictions
        try:
            if use_tta:
                # Use TTA
                self.logger.info("Making predictions with test-time augmentation")
                y_pred_proba = self.predict_with_tta(X_test)
            else:
                # Standard predict
                self.logger.info("Making standard predictions")
                y_pred_proba = self.predict_proba(X_test)
            
            # Check for NaN
            if np.isnan(y_pred_proba).any():
                self.logger.error("NaN values in predictions. Using default metrics.")
                return default_metrics
            
            # Get class predictions
            if self.binary:
                y_pred = (y_pred_proba > self.decision_threshold).astype(int)
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                y_true, y_pred, y_pred_proba, class_names, 
                visualize=visualize, detailed=detailed
            )
            
            # Log results
            self.logger.info("Evaluation completed successfully")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"{metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return default_metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, class_names=None, 
                          visualize=True, detailed=True):
        # Calculate metrics
        from sklearn.metrics import (
            precision_recall_curve, roc_curve, auc, confusion_matrix, 
            classification_report, precision_score, recall_score, f1_score,
            average_precision_score, roc_auc_score, accuracy_score
        )
        
        # Init metrics
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Binary metrics
        if self.binary:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm
            
            # Calculate PR metrics
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            
            # ROC metrics
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Specificity (TNR)
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Decision threshold
            metrics['threshold'] = self.decision_threshold
            
            # Classification report
            if class_names is None:
                class_names = ['Benign', 'Malignant']
                
            report = classification_report(y_true, y_pred, target_names=class_names)
            self.logger.info(f"\nClassification Report:\n{report}")
            
            # Confusion matrix
            self.logger.info("\nConfusion Matrix:")
            self.logger.info(f"TN: {tn}, FP: {fp}")
            self.logger.info(f"FN: {fn}, TP: {tp}")
            
            # Create visualizations
            if visualize:
                self._create_binary_visualizations(y_true, y_pred, y_pred_proba, class_names)
                
            # Detailed metrics
            if detailed:
                # Calibration curve
                try:
                    from sklearn.calibration import calibration_curve
                    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
                    metrics['calibration_true'] = prob_true.tolist()
                    metrics['calibration_pred'] = prob_pred.tolist()
                except:
                    self.logger.warning("Could not calculate calibration curve")
                
                # Additional metrics
                metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
                metrics['positive_predictive_value'] = metrics['precision']
                metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                # Optimal threshold
                precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_proba)
                f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
                best_f1_idx = np.argmax(f1_scores)
                metrics['optimal_f1_threshold'] = thresholds_pr[best_f1_idx] if best_f1_idx < len(thresholds_pr) else 0.5
                
                # Brier score
                try:
                    from sklearn.metrics import brier_score_loss
                    metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
                except:
                    self.logger.warning("Could not calculate Brier score")
        
        # Multi-class metrics
        else:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm
            
            # PR metrics
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            # ROC metrics
            if len(np.unique(y_true)) > 1:
                try:
                    # One-hot for multi-class
                    y_true_onehot = np.zeros((len(y_true), self.num_classes))
                    y_true_onehot[np.arange(len(y_true)), y_true] = 1
                    
                    metrics['auc'] = roc_auc_score(y_true_onehot, y_pred_proba, multi_class='ovr')
                except:
                    self.logger.warning("Could not calculate multi-class AUC")
                    metrics['auc'] = 0.5
            else:
                metrics['auc'] = 0.5
            
            # Classification report
            if class_names is None:
                class_names = [f"Class {i}" for i in range(self.num_classes)]
                
            report = classification_report(y_true, y_pred, target_names=class_names)
            self.logger.info(f"\nClassification Report:\n{report}")
            
            # Create visualizations
            if visualize:
                self._create_multiclass_visualizations(y_true, y_pred, y_pred_proba, class_names)
        
        return metrics
    
    def _create_binary_visualizations(self, y_true, y_pred, y_pred_proba, class_names):
        # Binary visualizations
        from sklearn.metrics import precision_recall_curve, roc_curve, auc
        
        # ROC Curve
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(self.model_dir, 'visualizations', 'roc_curve.png'), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating ROC curve: {e}")
        
        # PR Curve
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = auc(recall, precision)
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(True)
            plt.savefig(os.path.join(self.model_dir, 'visualizations', 'precision_recall_curve.png'), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating Precision-Recall curve: {e}")
        
        # Confusion Matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            
            # Add annotations
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join(self.model_dir, 'visualizations', 'confusion_matrix.png'), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating confusion matrix: {e}")
        
        # Prediction Distribution
        try:
            plt.figure(figsize=(10, 8))
            plt.hist(y_pred_proba[y_true == 0], bins=20, alpha=0.5, label=class_names[0])
            plt.hist(y_pred_proba[y_true == 1], bins=20, alpha=0.5, label=class_names[1])
            plt.axvline(x=self.decision_threshold, color='r', linestyle='--', 
                        label=f'Threshold = {self.decision_threshold:.3f}')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Count')
            plt.title('Distribution of Predicted Probabilities')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.model_dir, 'visualizations', 'prediction_distribution.png'), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating distribution plot: {e}")
        
        # Calibration curve
        try:
            from sklearn.calibration import calibration_curve
            
            plt.figure(figsize=(10, 8))
            prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
            
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
            
            plt.xlabel('Mean predicted probability')
            plt.ylabel('Fraction of positives')
            plt.title('Calibration Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.model_dir, 'visualizations', 'calibration_curve.png'), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating calibration curve: {e}")
    
    def _create_multiclass_visualizations(self, y_true, y_pred, y_pred_proba, class_names):
        # Multi-class visualizations
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        # Confusion Matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            
            # Add annotations
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join(self.model_dir, 'visualizations', 'confusion_matrix.png'), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating confusion matrix: {e}")
            
        # Class-wise ROC
        try:
            # One-hot for multi-class
            y_true_onehot = np.zeros((len(y_true), self.num_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            
            plt.figure(figsize=(10, 8))
            
            for i in range(self.num_classes):
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-class ROC Curve (One-vs-Rest)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.model_dir, 'visualizations', 'multiclass_roc.png'), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating multi-class ROC curves: {e}")
        
        # Class distribution
        try:
            plt.figure(figsize=(10, 8))
            
            # Count instances
            true_counts = np.bincount(y_true, minlength=self.num_classes)
            pred_counts = np.bincount(y_pred, minlength=self.num_classes)
            
            x = np.arange(len(class_names))
            width = 0.35
            
            plt.bar(x - width/2, true_counts, width, label='Actual')
            plt.bar(x + width/2, pred_counts, width, label='Predicted')
            
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Class Distribution')
            plt.xticks(x, class_names)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.model_dir, 'visualizations', 'class_distribution.png'), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating class distribution plot: {e}")
        
        # Confidence distribution
        try:
            plt.figure(figsize=(10, 8))
            
            # Max probability
            class_probs = np.max(y_pred_proba, axis=1)
            
            # Split by correctness
            correct = y_pred == y_true
            
            plt.hist(class_probs[correct], bins=20, alpha=0.5, label='Correct predictions')
            plt.hist(class_probs[~correct], bins=20, alpha=0.5, label='Incorrect predictions')
            
            plt.xlabel('Prediction Confidence')
            plt.ylabel('Count')
            plt.title('Distribution of Prediction Confidence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.model_dir, 'visualizations', 'confidence_distribution.png'), dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating confidence distribution plot: {e}")
    
    def evaluate_ensemble(self, X_test, y_test, class_names=None):
        # Evaluate ensemble
        if not self.models:
            self.logger.warning("No ensemble models available. Using single model.")
            return self.evaluate(X_test, y_test, class_names)
        
        # Get ensemble predictions
        y_pred_proba = self.predict_with_ensemble(X_test)
        
        # Get class predictions
        if self.binary:
            y_pred = (y_pred_proba > self.decision_threshold).astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Format labels
        if self.binary:
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_true = np.argmax(y_test, axis=1)
            else:
                y_true = y_test.flatten() if len(y_test.shape) > 1 else y_test
        else:
            # Get class indices
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_true = np.argmax(y_test, axis=1)
            else:
                y_true = y_test
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            y_true, y_pred, y_pred_proba, class_names, 
            visualize=True, detailed=True
        )
        
        # Add ensemble info
        metrics['num_models'] = len(self.models)
        
        # Optimize threshold
        if self.binary:
            optimal_threshold = self.optimize_threshold(X_test, y_test)
            metrics['ensemble_threshold'] = optimal_threshold
        
        return metrics
    
    def plot_training_history(self):
        # Plot training curves
        if self.history is None or not self.history:
            self.logger.warning("No training history available. Train the model first.")
            return
            
        # Check for NaN
        has_nan = False
        for metric, values in self.history.items():
            if any(np.isnan(val) if not isinstance(val, str) else val == 'nan' for val in values):
                has_nan = True
                self.logger.warning(f"NaN values detected in {metric}")
        
        if has_nan:
            self.logger.warning("Some metrics contain NaN values. Plotting will be limited.")
            
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Metrics to plot
        metrics_to_plot = [
            ('accuracy', 'val_accuracy', 'Model Accuracy', 'Accuracy', 'lower right'),
            ('loss', 'val_loss', 'Model Loss', 'Loss', 'upper right'),
            ('precision', 'val_precision', 'Model Precision', 'Precision', 'lower right'),
            ('recall', 'val_recall', 'Model Recall', 'Recall', 'lower right')
        ]
        
        # Plot each metric
        for i, (train_metric, val_metric, title, ylabel, legend_loc) in enumerate(metrics_to_plot):
            ax = axes[i // 2, i % 2]
            
            # Check if valid data
            if (train_metric in self.history and 
                not all(np.isnan(val) if not isinstance(val, str) else val == 'nan' 
                        for val in self.history[train_metric])):
                
                # Plot training
                ax.plot(self.history[train_metric], label=f'Train {ylabel}')
                
                # Plot validation
                if (val_metric in self.history and 
                    not all(np.isnan(val) if not isinstance(val, str) else val == 'nan' 
                            for val in self.history[val_metric])):
                    ax.plot(self.history[val_metric], label=f'Validation {ylabel}')
                
                # Add labels
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_xlabel('Epoch')
                ax.legend(loc=legend_loc)
                ax.grid(True)
                
                # Mark best epoch
                if val_metric in self.history:
                    # Convert to numeric
                    val_values = np.array([
                        float(val) if not np.isnan(val) and not isinstance(val, str) 
                        else np.nan for val in self.history[val_metric]
                    ])
                    
                    # Get best
                    if 'loss' in val_metric:
                        # Lower is better
                        best_epoch = np.nanargmin(val_values)
                        best_value = np.nanmin(val_values)
                    else:
                        # Higher is better
                        best_epoch = np.nanargmax(val_values)
                        best_value = np.nanmax(val_values)
                    
                    # Mark it
                    if not np.isnan(best_value):
                        ax.axvline(x=best_epoch, color='r', linestyle='--')
                        ax.text(
                            best_epoch, best_value,
                            f'  Best: {best_value:.4f}',
                            verticalalignment='center'
                        )
            else:
                # No data
                ax.text(
                    0.5, 0.5,
                    f'No valid {ylabel} data',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes
                )
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'visualizations', 'training_history.png'), dpi=300)
        plt.close()
        
        # Check overfitting
        if 'val_loss' in self.history:
            val_loss = self.history['val_loss']
            
            # Find best epoch
            valid_val_loss = [
                float(val) if not np.isnan(val) and not isinstance(val, str) 
                else np.inf for val in val_loss
            ]
            min_val_loss_epoch = np.argmin(valid_val_loss) + 1  # +1 for 1-indexed
            
            self.logger.info(f"Best validation loss at epoch {min_val_loss_epoch}")
            
            # Check for overfitting
            if min_val_loss_epoch < len(val_loss) - 5:
                # Calculate slope
                slope = (valid_val_loss[-1] - valid_val_loss[min_val_loss_epoch-1]) / (len(valid_val_loss) - min_val_loss_epoch)
                
                if slope > 0.01:
                    self.logger.warning(f"Significant overfitting detected after epoch {min_val_loss_epoch}")
                elif slope > 0.001:
                    self.logger.info("Mild overfitting detected after the best epoch")
                else:
                    self.logger.info("No clear overfitting detected")
            else:
                self.logger.info("No clear overfitting detected")
    
    def visualize_model_predictions(self, X, y_true, class_names=None, samples=10):
        # Show model predictions
        # Limit samples
        n_samples = min(samples, len(X))
        
        # Default class names
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        # Get predictions
        y_pred_proba = self.predict_proba(X[:n_samples])
        
        if self.binary:
            y_pred = (y_pred_proba > self.decision_threshold).astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Format labels
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Create image grid
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = [axes]
        
        # Show each sample
        for i in range(n_samples):
            ax = axes[i]
            img = X[i]
            
            # Convert for display
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            # Show image
            ax.imshow(img)
            
            # Get predictions
            true_label = y_true[i]
            pred_label = y_pred[i]
            
            if self.binary:
                pred_prob = y_pred_proba[i]
                prob_str = f"{pred_prob:.4f}"
            else:
                pred_prob = y_pred_proba[i, pred_label]
                prob_str = f"{pred_prob:.4f}"
            
            # Create title
            title = f"True: {class_names[true_label]}, Pred: {class_names[pred_label]} ({prob_str})"
            
            # Mark correct/incorrect
            if true_label == pred_label:
                title += " ✓"
                ax.set_title(title, color='green')
            else:
                title += " ✗"
                ax.set_title(title, color='red')
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'visualizations', 'model_predictions.png'), dpi=300)
        plt.close()
        
        self.logger.info(f"Visualized {n_samples} model predictions")
    
    def visualize_attention(self, X, samples=5):
        # Show attention maps
        if not self.use_attention:
            self.logger.warning("Model does not use attention. Cannot visualize.")
            return
        
        import cv2
        
        # Limit samples
        n_samples = min(samples, len(X))
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
        if n_samples == 1:
            axes = [axes]
        
        # Use Grad-CAM
        try:
            # Create visualization model
            from tensorflow.keras.models import Model
            
            # Find last conv layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                # Look for conv layer
                if 'conv' in layer.name.lower():
                    last_conv_layer = layer.name
                    break
                    
            if last_conv_layer is None:
                # Try blocks for EfficientNet
                for layer in reversed(self.model.layers):
                    if 'block' in layer.name.lower() and hasattr(layer, 'output_shape'):
                        last_conv_layer = layer.name
                        break
            
            if last_conv_layer is None:
                self.logger.warning("Could not find convolutional layer for attention visualization")
                return
                
            # Create model with both outputs
            grad_model = Model(
                inputs=self.model.inputs,
                outputs=[self.model.output, self.model.get_layer(last_conv_layer).output]
            )
            
            # Process each sample
            for i in range(n_samples):
                img = X[i:i+1]  # Add batch dim
                
                # Get activations
                with tf.GradientTape() as tape:
                    preds, conv_outputs = grad_model(img)
                    if self.binary:
                        class_idx = 1  # Malignant class
                        class_output = preds[:, 0]
                    else:
                        class_idx = tf.argmax(preds[0])
                        class_output = preds[:, class_idx]
                
                # Get gradients
                grads = tape.gradient(class_output, conv_outputs)
                
                # Pool gradients
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                
                # Weight channels
                conv_outputs = conv_outputs[0]
                heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
                
                # Normalize heatmap
                heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
                heatmap = heatmap.numpy()
                
                # Resize to match image
                heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
                
                # Convert to RGB
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Get display image
                display_img = img[0].copy()
                if display_img.dtype != np.uint8:
                    display_img = (display_img * 255).astype(np.uint8)
                
                # Fix BGR/RGB
                if display_img.shape[-1] == 3:
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Create overlay
                superimposed_img = heatmap * 0.4 + display_img
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                
                # Show original
                axes[i, 0].imshow(display_img)
                axes[i, 0].set_title("Original Image")
                axes[i, 0].axis('off')
                
                # Show heatmap
                axes[i, 1].imshow(heatmap)
                axes[i, 1].set_title("Attention Heatmap")
                axes[i, 1].axis('off')
                
                # Show overlay
                axes[i, 2].imshow(superimposed_img)
                axes[i, 2].set_title("Attention Overlay")
                axes[i, 2].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'visualizations', 'attention_maps.png'), dpi=300)
            plt.close()
            
            self.logger.info(f"Visualized attention for {n_samples} images")
                
        except Exception as e:
            self.logger.error(f"Error visualizing attention: {e}")
    

    def save_model(self, filepath=None, save_weights_only=False):
        # Save model to disk
        if filepath is None:
            # Default path
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(self.model_dir, 'checkpoints', f'leukemia_v5_{timestamp}.h5')
        
        try:
            # Define custom objects
            custom_objects = {
                'AttentionModule': AttentionModule,
                'FocalLoss': FocalLoss
            }
            
            if save_weights_only:
                # Just weights
                self.model.save_weights(filepath)
                self.logger.info(f"Model weights saved to {filepath}")
            else:
                # Create directory
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save weights
                weights_path = filepath.replace('.h5', '_weights.h5')
                self.model.save_weights(weights_path)
                
                # Save architecture
                json_path = filepath.replace('.h5', '_architecture.json')
                with open(json_path, 'w') as f:
                    f.write(self.model.to_json())
                
                # Save metadata
                metadata = {
                    'img_size': self.img_size,
                    'num_classes': self.num_classes,
                    'binary': self.binary,
                    'backbone': self.backbone,
                    'decision_threshold': float(self.decision_threshold),
                    'dropout_rate': self.dropout_rate,
                    'use_attention': self.use_attention,
                    'use_focal_loss': self.use_focal_loss,
                    'version': '5.0',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                metadata_path = filepath.replace('.h5', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                self.logger.info(f"Model architecture saved to {json_path}")
                self.logger.info(f"Model weights saved to {weights_path}")
                self.logger.info(f"Model metadata saved to {metadata_path}")
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return None

    def load_model(self, filepath, load_weights_only=False):
        # Load saved model
        try:
            # Define custom objects
            custom_objects = {
                'AttentionModule': AttentionModule,
                'FocalLoss': FocalLoss
            }
            
            if load_weights_only:
                # Just weights
                self.model.load_weights(filepath)
                self.logger.info(f"Model weights loaded from {filepath}")
            else:
                # Check model format
                weights_path = filepath.replace('.h5', '_weights.h5')
                json_path = filepath.replace('.h5', '_architecture.json')
                
                if os.path.exists(weights_path) and os.path.exists(json_path):
                    # Custom format
                    with open(json_path, 'r') as f:
                        model_json = f.read()
                    
                    # Load architecture
                    self.model = tf.keras.models.model_from_json(
                        model_json, 
                        custom_objects=custom_objects
                    )
                    
                    # Load weights
                    self.model.load_weights(weights_path)
                    self.logger.info(f"Model loaded from {json_path} and {weights_path}")
                else:
                    # Standard format
                    self.model = tf.keras.models.load_model(
                        filepath,
                        custom_objects=custom_objects
                    )
                    self.logger.info(f"Model loaded from {filepath}")
                
                # Load metadata
                metadata_path = filepath.replace('.h5', '_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Update params
                    self.img_size = tuple(metadata.get('img_size', self.img_size))
                    self.num_classes = metadata.get('num_classes', self.num_classes)
                    self.binary = metadata.get('binary', self.binary)
                    self.backbone = metadata.get('backbone', self.backbone)
                    self.decision_threshold = metadata.get('decision_threshold', self.decision_threshold)
                    self.dropout_rate = metadata.get('dropout_rate', self.dropout_rate)
                    self.use_attention = metadata.get('use_attention', self.use_attention)
                    self.use_focal_loss = metadata.get('use_focal_loss', self.use_focal_loss)
                    
                    self.logger.info(f"Model metadata loaded from {metadata_path}")
            
            # Recompile model
            self._compile_model(self.model)
            self.logger.info("Model recompiled with appropriate optimizer and loss")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def export_onnx(self, filepath=None):
        # Export to ONNX
        if filepath is None:
            # Default path
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(self.model_dir, 'exports', f'leukemia_v5_{timestamp}.onnx')
        
        try:
            import tf2onnx
            
            # Create directory
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert to ONNX
            tf2onnx.convert.from_keras(
                self.model,
                output_path=filepath
            )
            
            self.logger.info(f"Model exported to ONNX format at {filepath}")
            return filepath
            
        except ImportError:
            self.logger.error("tf2onnx not installed. Run 'pip install tf2onnx' to enable ONNX export.")
            return None
        except Exception as e:
            self.logger.error(f"Error exporting to ONNX: {e}")
            return None