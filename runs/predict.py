#!/usr/bin/env python

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import cv2
import glob
from datetime import datetime
import logging

# Import modules
from runs.model import LeukemiaClassifierV5
from runs.layers import AttentionModule, FocalLoss

def parse_args():
    # Setup CLI args
    parser = argparse.ArgumentParser(description='Leukemia Classifier V5 Prediction')
    
    # Input args
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (both width and height)')
    parser.add_argument('--output_dir', type=str, default='prediction_results',
                        help='Directory to save prediction results')
    
    # Model args
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.h5 file)')
    parser.add_argument('--binary', action='store_true', default=True,
                        help='Use binary classification (Benign vs ALL)')
    parser.add_argument('--class_names', type=str, nargs='+',
                        default=['Benign', 'Malignant'],
                        help='Class names for prediction')
    
    # Prediction args
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom threshold for binary classification')
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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for prediction')
    
    return parser.parse_args()

def load_and_preprocess_image(image_path, target_size=(224, 224), enhance=True):
    # Process single image
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image {image_path}")
    
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Enhance image
    if enhance:
        # RGB to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # CLAHE on L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L_channel = lab[..., 0].copy().astype(np.uint8)
        lab[..., 0] = clahe.apply(L_channel)
        
        # LAB to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Balance colors
        r, g, b = cv2.split(enhanced)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        r_clahe = clahe.apply(r.astype(np.uint8))
        g_clahe = clahe.apply(g.astype(np.uint8))
        b_clahe = clahe.apply(b.astype(np.uint8))
        balanced = cv2.merge([r_clahe, g_clahe, b_clahe])
        
        # Sharpen image
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) / 5.0
        sharpened = cv2.filter2D(balanced, -1, kernel)
        
        # Denoise if needed
        gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
        noise_level = np.std(gray) / np.mean(gray)
        
        if noise_level > 0.15:
            denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 5, 5, 7, 21)
            img = denoised
        else:
            img = sharpened
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize pixels
    img = img.astype(np.float32) / 255.0
    
    return img

def find_images(input_path):
    # Find all images
    if os.path.isfile(input_path):
        # Single file
        return [input_path]
    elif os.path.isdir(input_path):
        # Process directory
        image_extensions = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, f"*.{ext}")))
            image_files.extend(glob.glob(os.path.join(input_path, f"*.{ext.upper()}")))
            # Check subdirectories
            image_files.extend(glob.glob(os.path.join(input_path, "**", f"*.{ext}"), recursive=True))
            image_files.extend(glob.glob(os.path.join(input_path, "**", f"*.{ext.upper()}"), recursive=True))
        
        return sorted(list(set(image_files)))  # Remove duplicates
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

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
    # Main prediction function
    # Parse args
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('LeukemiaClassifierV5_Prediction')
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find images
    try:
        image_paths = find_images(args.input)
        if not image_paths:
            logger.error(f"No images found in {args.input}")
            return
        
        logger.info(f"Found {len(image_paths)} images for prediction")
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Load model
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
            num_classes=metadata.get('num_classes', len(args.class_names)),
            backbone=metadata.get('backbone', 'efficientnet'),
            model_dir=args.output_dir,
            binary=metadata.get('binary', args.binary),
            dropout_rate=metadata.get('dropout_rate', 0.5),
            use_attention=metadata.get('use_attention', True),
            use_focal_loss=metadata.get('use_focal_loss', True)
        )
        
        # Load class names
        if 'class_names' in metadata and metadata['class_names']:
            args.class_names = metadata['class_names']
            logger.info(f"Using class names from metadata: {args.class_names}")
        
        # Set threshold
        if 'decision_threshold' in metadata and args.threshold is None:
            classifier.decision_threshold = metadata['decision_threshold']
    else:
        # Create with defaults
        logger.info("No metadata found, creating model with default settings")
        classifier = LeukemiaClassifierV5(
            img_size=(args.img_size, args.img_size),
            num_classes=len(args.class_names),
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
    
    # Preprocess images
    logger.info("Loading and preprocessing images")
    preprocessed_images = []
    valid_image_paths = []
    
    for image_path in image_paths:
        try:
            preprocessed_img = load_and_preprocess_image(
                image_path, 
                target_size=(args.img_size, args.img_size),
                enhance=True
            )
            preprocessed_images.append(preprocessed_img)
            valid_image_paths.append(image_path)
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {e}")
    
    if not preprocessed_images:
        logger.error("No valid images found for prediction")
        return
    
    # Convert to array
    preprocessed_images = np.array(preprocessed_images)
    
    # Run predictions
    logger.info(f"Making predictions for {len(preprocessed_images)} images")
    
    # Set MC samples
    if args.use_mc_dropout:
        classifier.mc_samples = args.mc_samples
    
    results = []
    
    # Process in batches
    for i in range(0, len(preprocessed_images), args.batch_size):
        batch_images = preprocessed_images[i:i+args.batch_size]
        batch_paths = valid_image_paths[i:i+args.batch_size]
        
        # Make predictions
        if args.use_mc_dropout:
            # Use MC dropout
            y_pred_proba, y_pred, uncertainty = classifier.predict_with_uncertainty(batch_images)
        elif args.use_tta:
            # Use test-time aug
            y_pred_proba = classifier.predict_with_tta(batch_images, n_augmentations=args.n_tta)
            # Get class predictions
            if args.binary:
                y_pred = (y_pred_proba > classifier.decision_threshold).astype(int)
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
            uncertainty = None
        else:
            # Standard prediction
            y_pred_proba = classifier.predict_proba(batch_images)
            # Get class predictions
            if args.binary:
                y_pred = (y_pred_proba > classifier.decision_threshold).astype(int)
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
            uncertainty = None
        
        # Process results
        for j in range(len(batch_images)):
            result = {
                'image_path': batch_paths[j],
                'filename': os.path.basename(batch_paths[j])
            }
            
            if args.binary:
                # Binary case
                result['predicted_probability'] = float(y_pred_proba[j])
                result['predicted_class_idx'] = int(y_pred[j])
                result['predicted_class'] = args.class_names[y_pred[j]]
                
                if uncertainty is not None:
                    result['uncertainty'] = float(uncertainty[j])
            else:
                # Multi-class case
                result['predicted_probabilities'] = [float(p) for p in y_pred_proba[j]]
                result['predicted_class_idx'] = int(y_pred[j])
                result['predicted_class'] = args.class_names[y_pred[j]]
                
                if uncertainty is not None:
                    result['uncertainty'] = float(uncertainty[j])
            
            results.append(result)
    
    # Sort results
    if args.binary:
        if args.use_mc_dropout:
            # Sort by uncertainty
            results.sort(key=lambda x: x.get('uncertainty', 0))
        else:
            # Sort by probability
            results.sort(key=lambda x: x.get('predicted_probability', 0), reverse=True)
    else:
        if args.use_mc_dropout:
            # Sort by uncertainty
            results.sort(key=lambda x: x.get('uncertainty', 0))
        else:
            # Sort by max prob
            results.sort(key=lambda x: max(x.get('predicted_probabilities', [0])), reverse=True)
    
    # Save JSON results
    results_path = os.path.join(args.output_dir, 'prediction_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Saved prediction results to {results_path}")
    
    # Create visualizations
    if args.visualize:
        logger.info("Creating visualizations")
        
        # Create viz dir
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Limit images
        max_viz = min(20, len(results))
        
        # Create figure grid
        n_rows = (max_viz + 4) // 5  # 5 per row
        n_cols = min(5, max_viz)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        
        # Show predictions
        for i in range(max_viz):
            result = results[i]
            img_path = result['image_path']
            
            # Read original image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get grid position
            row = i // n_cols
            col = i % n_cols
            
            # Get correct axis
            if n_rows == 1:
                ax = axes[col]
            elif n_cols == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            # Show image
            ax.imshow(img)
            
            # Add prediction info
            if args.binary:
                # Binary info
                pred_class = result['predicted_class']
                pred_prob = result['predicted_probability']
                
                title = f"{pred_class}\nProb: {pred_prob:.4f}"
                
                # Add uncertainty
                if 'uncertainty' in result:
                    title += f"\nUncertainty: {result['uncertainty']:.4f}"
                
                # Color code
                title_color = 'green' if pred_class == args.class_names[0] else 'red'
                
            else:
                # Multi-class info
                pred_class = result['predicted_class']
                pred_probs = result['predicted_probabilities']
                max_prob = max(pred_probs)
                
                title = f"{pred_class}\nProb: {max_prob:.4f}"
                
                # Add uncertainty
                if 'uncertainty' in result:
                    title += f"\nUncertainty: {result['uncertainty']:.4f}"
                
                # Color code
                title_color = 'C' + str(result['predicted_class_idx'] % 10)
            
            ax.set_title(title, color=title_color)
            ax.axis('off')
        
        # Clear unused plots
        for i in range(max_viz, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows == 1:
                ax = axes[col]
            elif n_cols == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            ax.axis('off')
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'prediction_visualization.png'), dpi=300)
        plt.close()
        
        logger.info(f"Saved prediction visualization to {viz_dir}")
    
        # Binary histogram
        if args.binary:
            plt.figure(figsize=(10, 6))
            
            probabilities = [r['predicted_probability'] for r in results]
            bins = np.linspace(0, 1, 21)  # 20 bins
            
            plt.hist(probabilities, bins=bins, color='skyblue', edgecolor='black')
            plt.axvline(x=classifier.decision_threshold, color='red', linestyle='--', 
                        label=f'Threshold = {classifier.decision_threshold:.3f}')
            
            plt.xlabel('Predicted Probability')
            plt.ylabel('Count')
            plt.title('Distribution of Predicted Probabilities')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save histogram
            plt.savefig(os.path.join(viz_dir, 'probability_histogram.png'), dpi=300)
            plt.close()
            
            logger.info(f"Saved probability histogram to {viz_dir}")
        
        # Uncertainty histogram
        if args.use_mc_dropout:
            plt.figure(figsize=(10, 6))
            
            uncertainties = [r['uncertainty'] for r in results]
            
            plt.hist(uncertainties, bins=20, color='skyblue', edgecolor='black')
            
            plt.xlabel('Uncertainty')
            plt.ylabel('Count')
            plt.title('Distribution of Prediction Uncertainty')
            plt.grid(True, alpha=0.3)
            
            # Save histogram
            plt.savefig(os.path.join(viz_dir, 'uncertainty_histogram.png'), dpi=300)
            plt.close()
            
            logger.info(f"Saved uncertainty histogram to {viz_dir}")
    
    # Print summary
    logger.info("Prediction Summary:")
    
    if args.binary:
        # Count per class
        benign_count = sum(1 for r in results if r['predicted_class_idx'] == 0)
        malignant_count = sum(1 for r in results if r['predicted_class_idx'] == 1)
        
        logger.info(f"  Total images: {len(results)}")
        logger.info(f"  {args.class_names[0]}: {benign_count} ({benign_count/len(results)*100:.1f}%)")
        logger.info(f"  {args.class_names[1]}: {malignant_count} ({malignant_count/len(results)*100:.1f}%)")
    else:
        # Count per class
        class_counts = {}
        for r in results:
            class_idx = r['predicted_class_idx']
            class_name = r['predicted_class']
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        logger.info(f"  Total images: {len(results)}")
        for class_idx, count in sorted(class_counts.items()):
            logger.info(f"  {args.class_names[class_idx]}: {count} ({count/len(results)*100:.1f}%)")
    
    logger.info("LeukemiaClassifierV5 prediction completed successfully")

if __name__ == '__main__':
    main()