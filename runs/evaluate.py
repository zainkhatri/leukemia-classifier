#!/usr/bin/env python

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging

# Import modules
from runs.data_processing import LeukemiaDataProcessor
from runs.model import LeukemiaClassifierV5
from runs.layers import AttentionModule, FocalLoss

def parse_args():
    # Setup CLI args
    parser = argparse.ArgumentParser(description='Leukemia Classifier V5 Evaluation')
    
    # Dataset args
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the test dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (both width and height)')
    parser.add_argument('--binary', action='store_true', default=True,
                        help='Use binary classification (Benign vs ALL)')
    
    # Model args
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.h5 file)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    
    # Eval args
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--use_tta', action='store_true', default=False,
                        help='Use test-time augmentation')
    parser.add_argument('--n_tta', type=int, default=5,
                        help='Number of TTA augmentations')
    parser.add_argument('--use_mc_dropout', action='store_true', default=False,
                        help='Use Monte Carlo Dropout for uncertainty estimation')
    parser.add_argument('--mc_samples', type=int, default=30,
                        help='Number of Monte Carlo samples')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Create visualizations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Extra eval args
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom threshold for binary classification')
    parser.add_argument('--calibrate', action='store_true', default=False,
                        help='Calibrate model probabilities after evaluation')
    parser.add_argument('--visualize_attention', action='store_true', default=True,
                        help='Visualize attention maps for selected samples')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Save detailed predictions for all samples')
    
    return parser.parse_args()

def set_random_seed(seed):
    # Set reproducible seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_model_custom(filepath):
    # Load split model files
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
    # Main evaluation function
    # Parse args
    args = parse_args()
    
    # Set seed
    set_random_seed(args.seed)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('LeukemiaClassifierV5_Evaluation')
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Init data processor
    logger.info(f"Loading test data from {args.data_dir}")
    data_processor = LeukemiaDataProcessor(args.data_dir)
    
    # Load test data
    test_images, test_labels, class_names, patient_ids = data_processor.load_images(
        target_size=(args.img_size, args.img_size),
        normalize=True,
        enhance=True,
        verbose=True,
        binary=args.binary,
        cached=True
    )
    
    logger.info(f"Loaded {len(test_images)} test images from {len(class_names)} classes")
    logger.info(f"Class names: {class_names}")
    logger.info(f"Class distribution: {np.bincount(test_labels)}")
    
    # Init classifier
    logger.info(f"Loading model from {args.model_path}")
    
    # Check for metadata
    metadata_path = args.model_path.replace('.h5', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create from metadata
        logger.info(f"Creating model with settings from metadata")
        classifier = LeukemiaClassifierV5(
            img_size=tuple(metadata.get('img_size', (args.img_size, args.img_size))),
            num_classes=metadata.get('num_classes', len(class_names)),
            backbone=metadata.get('backbone', 'efficientnet'),
            model_dir=args.output_dir,
            binary=metadata.get('binary', args.binary),
            dropout_rate=metadata.get('dropout_rate', 0.5),
            use_attention=metadata.get('use_attention', True),
            use_focal_loss=metadata.get('use_focal_loss', True)
        )
        
        # Set threshold
        if 'decision_threshold' in metadata and args.threshold is None:
            classifier.decision_threshold = metadata['decision_threshold']
    else:
        # Create with defaults
        logger.info("No metadata found, creating model with default settings")
        classifier = LeukemiaClassifierV5(
            img_size=(args.img_size, args.img_size),
            num_classes=len(class_names),
            backbone='efficientnet',
            model_dir=args.output_dir,
            binary=args.binary
        )
    
    # Load model weights
    try:
        # Try custom loader
        weights_path = args.model_path.replace('.h5', '_weights.h5')
        json_path = args.model_path.replace('.h5', '_architecture.json')
        
        if os.path.exists(weights_path) and os.path.exists(json_path):
            # Load split files
            classifier.model = load_model_custom(args.model_path)
            # Recompile model
            classifier._compile_model(classifier.model)
            logger.info("Model loaded successfully using custom loader")
        else:
            # Try standard loader
            custom_objects = {
                'AttentionModule': AttentionModule,
                'FocalLoss': FocalLoss
            }
            classifier.model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
            logger.info("Model loaded successfully with standard loader")
        
        success = True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        success = False
    
    if not success:
        logger.error(f"Failed to load model from {args.model_path}")
        return
    
    # Set custom threshold
    if args.threshold is not None:
        logger.info(f"Using custom threshold: {args.threshold}")
        classifier.decision_threshold = args.threshold
    
    # Run evaluation
    logger.info("Evaluating model on test data")
    
    # Standard eval
    metrics = classifier.evaluate(
        test_images, test_labels, class_names,
        use_tta=args.use_tta,
        visualize=args.visualize,
        detailed=True
    )
    
    # Log results
    logger.info("Evaluation metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (int, float, np.float32)) else v 
                for k, v in metrics.items() if k != 'confusion_matrix'}, f, indent=4)
    
    # Uncertainty eval (optional)
    if args.use_mc_dropout:
        logger.info(f"Evaluating uncertainty with Monte Carlo Dropout (samples={args.mc_samples})")
        
        # Set samples
        classifier.mc_samples = args.mc_samples
        
        # Get MC predictions
        y_pred_mean, y_pred_class, uncertainty = classifier.predict_with_uncertainty(test_images)
        
        # Calculate uncertainty metrics
        if args.binary:
            # Binary metrics
            uncertainty_metrics = {
                'mean_uncertainty': float(np.mean(uncertainty)),
                'median_uncertainty': float(np.median(uncertainty)),
                'max_uncertainty': float(np.max(uncertainty)),
                'min_uncertainty': float(np.min(uncertainty))
            }
            
            # Split by correctness
            y_true = test_labels
            if len(y_true.shape) > 1:
                y_true = np.argmax(y_true, axis=1)
            
            correct = y_pred_class == y_true
            uncertainty_correct = uncertainty[correct]
            uncertainty_incorrect = uncertainty[~correct]
            
            uncertainty_metrics['mean_uncertainty_correct'] = float(np.mean(uncertainty_correct) if len(uncertainty_correct) > 0 else 0)
            uncertainty_metrics['mean_uncertainty_incorrect'] = float(np.mean(uncertainty_incorrect) if len(uncertainty_incorrect) > 0 else 0)
            
            # Check pattern
            uncertainty_metrics['higher_uncertainty_for_incorrect'] = uncertainty_metrics['mean_uncertainty_incorrect'] > uncertainty_metrics['mean_uncertainty_correct']
        else:
            # Multi-class metrics
            uncertainty_metrics = {
                'mean_entropy': float(np.mean(uncertainty)),
                'median_entropy': float(np.median(uncertainty)),
                'max_entropy': float(np.max(uncertainty)),
                'min_entropy': float(np.min(uncertainty))
            }
            
            # Split by correctness
            y_true = test_labels
            if len(y_true.shape) > 1:
                y_true = np.argmax(y_true, axis=1)
            
            correct = y_pred_class == y_true
            uncertainty_correct = uncertainty[correct]
            uncertainty_incorrect = uncertainty[~correct]
            
            uncertainty_metrics['mean_entropy_correct'] = float(np.mean(uncertainty_correct) if len(uncertainty_correct) > 0 else 0)
            uncertainty_metrics['mean_entropy_incorrect'] = float(np.mean(uncertainty_incorrect) if len(uncertainty_incorrect) > 0 else 0)
            
            # Check pattern
            uncertainty_metrics['higher_uncertainty_for_incorrect'] = uncertainty_metrics['mean_entropy_incorrect'] > uncertainty_metrics['mean_entropy_correct']
        
        # Log results
        logger.info("Uncertainty metrics:")
        for metric, value in uncertainty_metrics.items():
            logger.info(f"  {metric}: {value}")
        
        # Save metrics
        uncertainty_path = os.path.join(args.output_dir, 'uncertainty_metrics.json')
        with open(uncertainty_path, 'w') as f:
            json.dump(uncertainty_metrics, f, indent=4)
        logger.info(f"Saved uncertainty metrics to {uncertainty_path}")
        
        # Visualize uncertainty
        if args.visualize:
            # Create viz dir
            uncertainty_dir = os.path.join(args.output_dir, 'uncertainty_viz')
            os.makedirs(uncertainty_dir, exist_ok=True)
            
            # Sample by uncertainty
            def sample_by_uncertainty(indices, uncertainty_values, n_samples=5):
                # Sample across range
                sorted_indices = np.argsort(uncertainty_values[indices])
                interval = max(1, len(sorted_indices) // n_samples)
                return [indices[sorted_indices[i * interval]] for i in range(min(n_samples, len(sorted_indices)))]
            
            y_true = test_labels
            if len(y_true.shape) > 1:
                y_true = np.argmax(y_true, axis=1)
            
            # Get prediction groups
            correct_indices = np.where(y_pred_class == y_true)[0]
            incorrect_indices = np.where(y_pred_class != y_true)[0]
            
            # Sample from groups
            correct_samples = sample_by_uncertainty(correct_indices, uncertainty)
            incorrect_samples = sample_by_uncertainty(incorrect_indices, uncertainty)
            
            # Create plot
            fig, axes = plt.subplots(2, len(correct_samples), figsize=(15, 10))
            
            # Plot correct preds
            for i, idx in enumerate(correct_samples):
                # Show image
                axes[0, i].imshow(test_images[idx])
                
                # Get pred info
                true_label = class_names[y_true[idx]]
                pred_label = class_names[y_pred_class[idx]]
                
                # Create title
                title = f"True: {true_label}\nPred: {pred_label}\nUncertainty: {uncertainty[idx]:.4f}"
                axes[0, i].set_title(title, color='green', fontsize=10)
                axes[0, i].axis('off')
            
            # Plot incorrect preds
            if len(incorrect_samples) > 0:
                for i, idx in enumerate(incorrect_samples):
                    if i < len(correct_samples):  # Check bounds
                        # Show image
                        axes[1, i].imshow(test_images[idx])
                        
                        # Get pred info
                        true_label = class_names[y_true[idx]]
                        pred_label = class_names[y_pred_class[idx]]
                        
                        # Create title
                        title = f"True: {true_label}\nPred: {pred_label}\nUncertainty: {uncertainty[idx]:.4f}"
                        axes[1, i].set_title(title, color='red', fontsize=10)
                        axes[1, i].axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(uncertainty_dir, 'uncertainty_examples.png'), dpi=300)
            plt.close()
            
            # Plot distributions
            plt.figure(figsize=(10, 6))
            
            if len(uncertainty_correct) > 0:
                plt.hist(uncertainty_correct, bins=20, alpha=0.6, label='Correct predictions')
            if len(uncertainty_incorrect) > 0:
                plt.hist(uncertainty_incorrect, bins=20, alpha=0.6, label='Incorrect predictions')
            
            plt.xlabel('Uncertainty')
            plt.ylabel('Count')
            plt.title('Uncertainty Distribution by Prediction Correctness')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(uncertainty_dir, 'uncertainty_distribution.png'), dpi=300)
            plt.close()
            
            logger.info(f"Saved uncertainty visualizations to {uncertainty_dir}")
    
    # Visualize attention maps
    if args.visualize_attention:
        logger.info("Visualizing attention maps")
        
        # Sample for viz
        n_samples = 5
        
        if args.binary:
            # Get balanced samples
            benign_indices = np.where(test_labels == 0)[0]
            malignant_indices = np.where(test_labels == 1)[0]
            
            # Sample per class
            benign_samples = np.random.choice(benign_indices, size=min(n_samples, len(benign_indices)), replace=False)
            malignant_samples = np.random.choice(malignant_indices, size=min(n_samples, len(malignant_indices)), replace=False)
            
            # Combine samples
            attention_samples = np.concatenate([benign_samples, malignant_samples])
        else:
            # Multi-class sampling
            attention_samples = []
            for class_idx in range(len(class_names)):
                class_indices = np.where(test_labels == class_idx)[0]
                if len(class_indices) > 0:
                    samples = np.random.choice(class_indices, size=min(n_samples, len(class_indices)), replace=False)
                    attention_samples.extend(samples)
        
        # Generate attention maps
        classifier.visualize_attention(test_images[attention_samples], samples=len(attention_samples))
    
    # Save detailed predictions
    if args.save_predictions:
        logger.info("Saving detailed predictions")
        
        # Create output dir
        predictions_dir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Get predictions
        y_pred_proba = classifier.predict_proba(test_images)
        y_pred = classifier.predict(test_images)
        
        # Format data
        y_true = test_labels
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Create prediction list
        predictions = []
        for i in range(len(test_images)):
            prediction = {
                'sample_idx': int(i),
                'true_label': int(y_true[i]),
                'true_class': class_names[y_true[i]]
            }
            
            if args.binary:
                prediction['predicted_probability'] = float(y_pred_proba[i])
                prediction['predicted_label'] = int(y_pred[i])
                prediction['predicted_class'] = class_names[y_pred[i]]
                prediction['correct'] = bool(y_pred[i] == y_true[i])
            else:
                prediction['predicted_probabilities'] = [float(p) for p in y_pred_proba[i]]
                prediction['predicted_label'] = int(y_pred[i])
                prediction['predicted_class'] = class_names[y_pred[i]]
                prediction['correct'] = bool(y_pred[i] == y_true[i])
            
            predictions.append(prediction)
        
        # Save to JSON
        predictions_path = os.path.join(predictions_dir, 'detailed_predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=4)
        logger.info(f"Saved detailed predictions to {predictions_path}")
    
    logger.info("LeukemiaClassifierV5 evaluation completed successfully")

if __name__ == '__main__':
    main()