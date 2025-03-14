#!/usr/bin/env python

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Import modules
from runs.data_processing import LeukemiaDataProcessor
from runs.model import LeukemiaClassifierV5
from runs.layers import AttentionModule, FocalLoss

def parse_args():
    # Setup CLI args
    parser = argparse.ArgumentParser(description='Leukemia Classifier V5 Training')
    
    # Dataset args
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (both width and height)')
    parser.add_argument('--binary', action='store_true', default=True,
                        help='Use binary classification (Benign vs ALL)')
    parser.add_argument('--cross_val', action='store_true', default=False,
                        help='Use cross-validation instead of train/val/test split')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--patient_aware', action='store_true', default=True,
                        help='Use patient-aware data splitting')
    
    # Model args
    parser.add_argument('--backbone', type=str, choices=['efficientnet', 'resnet'], 
                        default='efficientnet',
                        help='Model backbone architecture')
    parser.add_argument('--model_dir', type=str, default='models/v5',
                        help='Directory to save model checkpoints')
    parser.add_argument('--attention', action='store_true', default=True,
                        help='Use attention mechanism')
    parser.add_argument('--focal_loss', action='store_true', default=True,
                        help='Use focal loss for class imbalance')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for uncertainty estimation')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Initial learning rate')
    parser.add_argument('--class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced data')
    parser.add_argument('--augmentation', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--mixup', action='store_true', default=True,
                        help='Use mixup/cutmix augmentation')
    parser.add_argument('--fine_tuning', action='store_true', default=True,
                        help='Use gradual fine-tuning')
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='Train an ensemble of models')
    parser.add_argument('--n_models', type=int, default=3,
                        help='Number of models in the ensemble')
    
    # Eval args
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='Evaluate the model after training')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Create visualizations')
    
    # Other args
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Enable verbose output')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pre-trained model')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='Save the model after training')
    
    return parser.parse_args()

def set_random_seed(seed):
    # Set reproducible seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def custom_model_save(model, filepath):
    # Save model files
    # Create directory
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save weights
    weights_path = filepath.replace('.h5', '_weights.h5')
    model.save_weights(weights_path)
    
    # Save architecture
    json_path = filepath.replace('.h5', '_architecture.json')
    with open(json_path, 'w') as f:
        f.write(model.to_json())
    
    # Save layers code
    custom_objects_path = filepath.replace('.h5', '_custom_objects.py')
    with open(custom_objects_path, 'w') as f:
        f.write("""
from layers import AttentionModule, FocalLoss

def get_custom_objects():
    return {
        'AttentionModule': AttentionModule,
        'FocalLoss': FocalLoss
    }
""")
    
    return weights_path, json_path, custom_objects_path

def custom_model_load(filepath):
    # Load split model files
    # Check files exist
    weights_path = filepath.replace('.h5', '_weights.h5')
    json_path = filepath.replace('.h5', '_architecture.json')
    
    if not os.path.exists(weights_path) or not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find model files: {weights_path} or {json_path}")
    
    # Define custom layers
    custom_objects = {
        'AttentionModule': AttentionModule,
        'FocalLoss': FocalLoss
    }
    
    # Load from JSON
    with open(json_path, 'r') as f:
        model_json = f.read()
    
    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
    
    # Load weights
    model.load_weights(weights_path)
    
    return model

def main():
    # Main training function
    # Parse args
    args = parse_args()
    
    # Set seed
    set_random_seed(args.seed)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('LeukemiaClassifierV5')
    
    # Log params
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    data_processor = LeukemiaDataProcessor(args.data_dir)
    
    # Process images
    images, labels, class_names, patient_ids = data_processor.load_images(
        target_size=(args.img_size, args.img_size),
        normalize=True,
        enhance=True,
        verbose=args.verbose,
        binary=args.binary,
        cached=True  # Use cache
    )
    
    logger.info(f"Loaded {len(images)} images from {len(class_names)} classes")
    logger.info(f"Class names: {class_names}")
    logger.info(f"Class distribution: {np.bincount(labels)}")
    
    # Init classifier
    classifier = LeukemiaClassifierV5(
        img_size=(args.img_size, args.img_size),
        num_classes=len(class_names),
        backbone=args.backbone,
        model_dir=args.model_dir,
        binary=args.binary,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        use_attention=args.attention,
        use_focal_loss=args.focal_loss
    )
    
    # Load pretrained
    if args.load_model:
        logger.info(f"Loading pre-trained model from {args.load_model}")
        try:
            # Try custom loader
            classifier.model = custom_model_load(args.load_model)
            logger.info("Model loaded successfully using custom loader")
        except Exception as e:
            logger.error(f"Error loading model with custom loader: {e}")
            logger.info("Trying standard model loading...")
            
            try:
                # Try standard loader
                custom_objects = {
                    'AttentionModule': AttentionModule,
                    'FocalLoss': FocalLoss
                }
                classifier.model = tf.keras.models.load_model(args.load_model, custom_objects=custom_objects)
                logger.info("Model loaded successfully with standard loader")
            except Exception as e:
                logger.error(f"Error loading model with standard loader: {e}")
                logger.error("Could not load pre-trained model. Training from scratch.")
    
    # Training approach
    if args.cross_val:
        # Use cross-validation
        logger.info(f"Using {args.n_folds}-fold cross-validation")
        
        if args.ensemble:
            # Train ensemble CV
            cv_results, ensemble_models = classifier.train_with_kfold(
                images, labels,
                n_folds=args.n_folds,
                batch_size=args.batch_size,
                epochs=args.epochs,
                use_class_weights=args.class_weights,
                use_augmentation=args.augmentation,
                fine_tuning=args.fine_tuning
            )
            
            logger.info(f"Cross-validation results: {cv_results}")
            
            # Save ensemble
            if args.save_model:
                logger.info("Saving ensemble models")
                ensemble_dir = os.path.join(args.model_dir, 'ensemble')
                os.makedirs(ensemble_dir, exist_ok=True)
                
                for i, model in enumerate(ensemble_models):
                    # Save each model
                    model_base_path = os.path.join(ensemble_dir, f'model_{i+1}.h5')
                    weights_path, json_path, _ = custom_model_save(model, model_base_path)
                    logger.info(f"Ensemble model {i+1} saved to {model_base_path}")
            
            # Evaluate ensemble
            if args.evaluate:
                logger.info("Evaluating ensemble performance")
                ensemble_metrics = classifier.evaluate_ensemble(
                    images, labels, class_names
                )
                logger.info(f"Ensemble evaluation metrics: {ensemble_metrics}")
        else:
            # Regular CV
            cv_results, models = classifier.train_with_kfold(
                images, labels,
                n_folds=args.n_folds,
                batch_size=args.batch_size,
                epochs=args.epochs,
                use_class_weights=args.class_weights,
                use_augmentation=args.augmentation,
                fine_tuning=args.fine_tuning
            )
            
            logger.info(f"Cross-validation results: {cv_results}")
            
    else:
        # Use train/val/test
        logger.info("Using train/val/test split")
        
        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.create_data_splits(
            images, labels, patient_ids,
            test_size=0.15,
            val_size=0.15,
            stratify=True,
            patient_aware=args.patient_aware
        )
        
        if args.ensemble:
            # Train ensemble
            logger.info(f"Training ensemble of {args.n_models} models")
            
            models, ensemble_metrics = classifier.train_ensemble(
                X_train, X_val, y_train, y_val,
                n_models=args.n_models,
                batch_size=args.batch_size,
                epochs=args.epochs,
                use_augmentation=args.augmentation
            )
            
            logger.info(f"Ensemble training completed")
            logger.info(f"Ensemble validation metrics: {ensemble_metrics}")
            
            # Test ensemble
            if args.evaluate:
                logger.info("Evaluating ensemble on test set")
                test_metrics = classifier.evaluate_ensemble(
                    X_test, y_test, class_names
                )
                logger.info(f"Ensemble test metrics: {test_metrics}")
        else:
            # Train single model
            logger.info("Training a single model")
            
            history, metrics = classifier.train_single_model(
                X_train, X_val, y_train, y_val,
                batch_size=args.batch_size,
                epochs=args.epochs,
                use_class_weights=args.class_weights,
                use_augmentation=args.augmentation,
                use_mixup=args.mixup,
                fine_tuning=args.fine_tuning
            )
            
            logger.info(f"Training completed")
            logger.info(f"Validation metrics: {metrics}")
            
            # Save model
            if args.save_model:
                logger.info("Saving trained model")
                
                # Save model files
                model_path = os.path.join(args.model_dir, 'checkpoints', 'final_model.h5')
                weights_path, json_path, _ = custom_model_save(classifier.model, model_path)
                logger.info(f"Model architecture saved to {json_path}")
                logger.info(f"Model weights saved to {weights_path}")
                
                # Save metadata
                import json
                metadata = {
                    'img_size': list(classifier.img_size),
                    'num_classes': classifier.num_classes,
                    'binary': classifier.binary,
                    'backbone': classifier.backbone,
                    'decision_threshold': float(classifier.decision_threshold),
                    'dropout_rate': classifier.dropout_rate,
                    'use_attention': classifier.use_attention,
                    'use_focal_loss': classifier.use_focal_loss,
                    'version': '5.0',
                    'class_names': class_names.tolist() if isinstance(class_names, np.ndarray) else class_names,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                metadata_path = model_path.replace('.h5', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                logger.info(f"Model metadata saved to {metadata_path}")
            
            # Evaluate model
            if args.evaluate:
                logger.info("Evaluating on test set")
                test_metrics = classifier.evaluate(
                    X_test, y_test, class_names, 
                    use_tta=False,
                    visualize=args.visualize
                )
                logger.info(f"Test metrics: {test_metrics}")
            
            # Create visualizations
            if args.visualize:
                logger.info("Creating visualizations")
                
                # Show predictions
                classifier.visualize_model_predictions(
                    X_test[:10], y_test[:10], class_names,
                    samples=10
                )
                
                # Show attention maps
                if args.attention:
                    classifier.visualize_attention(
                        X_test[:5]
                    )
    
    logger.info("LeukemiaClassifierV5 training completed successfully")

if __name__ == '__main__':
    main()