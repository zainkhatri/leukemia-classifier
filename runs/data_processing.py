import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import re
import glob
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from skimage import exposure, transform, filters, feature
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from scipy import ndimage

class LeukemiaDataProcessor:
    # Enhanced data processor
    def __init__(self, data_dir, cache_dir='cache'):
        # Init with dirs
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.class_mapping = {}
        self.patient_mapping = {}
        self.augmenters = self._create_augmentation_pipeline()
        
        # Create cache dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _create_augmentation_pipeline(self):
        # Create aug pipelines
        # Standard aug
        standard_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.Affine(
                rotate=(-20, 20),
                scale=(0.85, 1.15),
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}
            ),
            iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.LinearContrast((0.8, 1.2)),
            iaa.AddToBrightness((-30, 30)),
            iaa.Sometimes(0.2, iaa.CLAHE(clip_limit=(1, 4)))
        ])
        
        # Strong synthetic aug
        synthetic_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                rotate=(-30, 30),
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
            ),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0))),
            iaa.LinearContrast((0.7, 1.3)),
            iaa.AddToBrightness((-40, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.MultiplySaturation((0.8, 1.2)),
            iaa.Sometimes(0.3, iaa.CLAHE(clip_limit=(1, 6))),
            iaa.Sometimes(0.2, iaa.ElasticTransformation(alpha=(0, 40), sigma=5)),
            iaa.Sometimes(0.4, iaa.PiecewiseAffine(scale=(0.01, 0.05)))
        ])
        
        # Val aug (conservative)
        val_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(
                rotate=(-10, 10),
                scale=(0.9, 1.1)
            ),
            iaa.LinearContrast((0.9, 1.1))
        ])
        
        # Test-time aug
        tta_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(
                rotate=[-15, 0, 15],
                scale=[0.9, 1, 1.1]
            ),
            iaa.LinearContrast([0.9, 1, 1.1])
        ])
        
        return {
            'standard': standard_aug,
            'synthetic': synthetic_aug,
            'validation': val_aug,
            'tta': tta_aug
        }
    
    def load_images(self, target_size=(224, 224), normalize=True, enhance=True, verbose=True, binary=True, cached=True):
        # Load dataset images
        cache_file = os.path.join(self.cache_dir, f"processed_images_{target_size[0]}x{target_size[1]}_{'binary' if binary else 'multi'}.npz")
        
        # Try cache first
        if cached and os.path.exists(cache_file):
            if verbose:
                print(f"Loading processed images from cache: {cache_file}")
            
            try:
                data = np.load(cache_file, allow_pickle=True)
                images = data['images']
                labels = data['labels']
                class_names = data['class_names']
                patient_ids = data['patient_ids']
                self.class_mapping = data['class_mapping'].item()
                
                if verbose:
                    print(f"Loaded {len(images)} images from cache")
                    class_counts = np.bincount(labels)
                    print(f"Class distribution: {class_counts}")
                
                self.patient_ids = patient_ids
                return images, labels, class_names, patient_ids
            except Exception as e:
                print(f"Error loading cache: {e}")
                print("Proceeding with regular image loading")
        
        # Check dir exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory {self.data_dir} not found")
        
        # Find all classes
        class_dirs = [d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))]
        
        if not class_dirs:
            raise ValueError(f"No subdirectories found in {self.data_dir}")
        
        # Debug paths
        if verbose:
            print(f"Absolute data directory path: {os.path.abspath(self.data_dir)}")
            for d in class_dirs:
                print(f"Found class directory: {os.path.abspath(os.path.join(self.data_dir, d))}")
        
        # Map class names
        self.class_mapping = {class_name: i for i, class_name in enumerate(sorted(class_dirs))}
        
        # Init lists
        images = []
        labels = []
        patient_ids = []
        class_image_counts = {}
        
        # Log classes
        if verbose:
            print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}")
        
        # Process each class
        for class_name in class_dirs:
            class_path = os.path.join(self.data_dir, class_name)
            
            # Binary mapping
            if binary:
                class_idx = 0 if 'Benign' in class_name else 1
            else:
                class_idx = self.class_mapping[class_name]
            
            # Get image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                image_files.extend(glob.glob(os.path.join(class_path, ext)))
                image_files.extend(glob.glob(os.path.join(class_path, "**", ext), recursive=True))
            
            if verbose:
                print(f"Loading {len(image_files)} images from class '{class_name}'")
                
            # Track counts
            class_image_counts[class_name] = len(image_files)
            
            # Process each image
            for img_path in tqdm(image_files, disable=not verbose):
                try:
                    # Get patient ID
                    patient_id = self._extract_patient_id(img_path)
                    
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                        
                    # BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Preprocess image
                    img = self._preprocess_image(img, enhance=enhance)
                    
                    # Resize image
                    img = cv2.resize(img, target_size)
                    
                    # Normalize if needed
                    if normalize:
                        img = img.astype(np.float32) / 255.0
                    
                    # Add to lists
                    images.append(img)
                    labels.append(class_idx)
                    patient_ids.append(patient_id)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Convert to arrays
        if len(images) == 0:
            raise ValueError("No valid images found in the dataset directories.")
            
        images = np.array(images)
        labels = np.array(labels)
        patient_ids = np.array(patient_ids)
        
        # Store patient IDs
        self.patient_ids = patient_ids
        
        if verbose:
            print(f"Loaded {len(images)} images total")
            class_counts = np.bincount(labels)
            print(f"Class distribution: {class_counts}")

        # Handle missing classes
        if binary:
            class_names = ["Benign", "Malignant"]
            
            # Check binary classes
            if len(np.unique(labels)) < 2:
                missing_class = 1 if 0 in labels else 0
                print(f"WARNING: Class {missing_class} not found in the dataset")
                print("Creating synthetic samples for the missing class")
                
                # Create synthetic samples
                existing_class = 1 - missing_class
                existing_images = images[labels == existing_class]
                
                # Calculate synth count
                maj_class_count = len(existing_images)
                num_synthetic = int(maj_class_count * 0.4)
                
                print(f"Creating {num_synthetic} synthetic samples for class {missing_class}")
                synthetic_images, synthetic_labels, synthetic_patient_ids = self._create_synthetic_samples_advanced(
                    existing_images, missing_class, num_synthetic
                )
                
                # Add synthetics
                if len(synthetic_images) > 0:
                    images = np.concatenate([images, synthetic_images])
                    labels = np.concatenate([labels, synthetic_labels])
                    patient_ids = np.concatenate([patient_ids, synthetic_patient_ids])
                else:
                    print("WARNING: Failed to create synthetic samples. Running with imbalanced dataset.")
                
                # Print updated stats
                print(f"Updated class distribution: {np.bincount(labels)}")
                
        else:
            class_names = list(self.class_mapping.keys())
            
            # Check multi classes
            unique_classes = np.unique(labels)
            if len(unique_classes) < len(class_names):
                print(f"WARNING: Only {len(unique_classes)} classes detected in the dataset")
                print(f"Expected {len(class_names)} classes")
                
                for class_idx in range(len(class_names)):
                    if class_idx not in unique_classes:
                        print(f"Creating synthetic samples for missing class {class_names[class_idx]}")
                        
                        # Create synthetic samples
                        num_synthetic = min(100, len(images) // (len(class_names) * 2))
                        synthetic_images, synthetic_labels, synthetic_patient_ids = self._create_synthetic_samples_advanced(
                            images, class_idx, num_synthetic
                        )
                        
                        # Add synthetics
                        if len(synthetic_images) > 0:
                            images = np.concatenate([images, synthetic_images])
                            labels = np.concatenate([labels, synthetic_labels])
                            patient_ids = np.concatenate([patient_ids, synthetic_patient_ids])
                
                # Print updated stats
                print(f"Updated class distribution: {np.bincount(labels, minlength=len(class_names))}")
        
        # Cache the data
        if cached:
            try:
                print(f"Caching processed images to {cache_file}")
                np.savez_compressed(
                    cache_file,
                    images=images,
                    labels=labels,
                    class_names=class_names,
                    patient_ids=patient_ids,
                    class_mapping=self.class_mapping
                )
            except Exception as e:
                print(f"Error caching images: {e}")
        
        return images, labels, class_names, patient_ids
    
    def _preprocess_image(self, img, enhance=True):
        # Advanced image preprocessing
        try:
            if not enhance:
                return img
            
            # RGB to LAB
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            
            # CLAHE on L
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            L_channel = lab[..., 0].copy().astype(np.uint8)
            lab[..., 0] = clahe.apply(L_channel)
            
            # LAB to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Color balance
            balanced = self._color_balance(enhanced)
            
            # Sharpen image
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]]) / 5.0
            sharpened = cv2.filter2D(balanced, -1, kernel)
            
            # Denoise if needed
            gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
            noise_level = np.std(gray) / np.mean(gray)
            
            if noise_level > 0.15:  # High noise detected
                denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 5, 5, 7, 21)
            else:
                denoised = sharpened
            
            return denoised
            
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return img  # Return original
    
    def _color_balance(self, img):
        # Balance color channels
        r, g, b = cv2.split(img)
        
        # CLAHE on each
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        r_clahe = clahe.apply(r.astype(np.uint8))
        g_clahe = clahe.apply(g.astype(np.uint8))
        b_clahe = clahe.apply(b.astype(np.uint8))
        
        # Merge channels
        balanced = cv2.merge([r_clahe, g_clahe, b_clahe])
        
        return balanced
    
    def _feature_extraction(self, img):
        # Extract image features
        # Convert to gray
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        # Ensure uint8
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        features = {}
        
        # Edge features
        edges = feature.canny(gray, sigma=1.0)
        features['edge_density'] = np.mean(edges)
        
        # Texture features
        try:
            from skimage.feature import graycomatrix, graycoprops
            glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
            features['contrast'] = np.mean(graycoprops(glcm, 'contrast'))
            features['homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            features['energy'] = np.mean(graycoprops(glcm, 'energy'))
        except:
            # Fallback metrics
            features['contrast'] = np.std(gray)
            features['homogeneity'] = 1.0 / (1.0 + features['contrast'])
            features['energy'] = np.sum(gray**2) / (gray.size * 255**2)
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        features['hist_mean'] = np.sum(np.arange(256) * hist_norm.flatten())
        features['hist_std'] = np.sqrt(np.sum(((np.arange(256) - features['hist_mean'])**2) * hist_norm.flatten()))
        
        # Brightness features
        features['brightness'] = np.mean(gray) / 255.0
        features['local_contrast'] = np.std(gray) / 255.0
        
        return features
    
    def _create_synthetic_samples_advanced(self, base_images, target_class, num_samples):
        # Create synthetic samples
        if len(base_images) == 0:
            print("ERROR: No base images provided for synthetic sample generation")
            return np.array([]), np.array([]), np.array([])
        
        synthetic_images = []
        synthetic_labels = []
        synthetic_patient_ids = []
        
        # Use strong aug
        augmenter = self.augmenters['synthetic']
        
        # Create samples
        indices = np.random.choice(len(base_images), num_samples, replace=True)
        
        for i, idx in enumerate(indices):
            try:
                # Get base image
                img = base_images[idx].copy()
                
                # Extract features
                features = self._feature_extraction(img)
                
                # Prep for aug
                if img.dtype != np.uint8 and np.max(img) <= 1.0:
                    img_aug = (img * 255).astype(np.uint8)
                else:
                    img_aug = img.copy()
                
                # Class-specific aug
                if target_class == 1:  # Malignant class
                    # Adjust based on features
                    if features['edge_density'] < 0.05:
                        # Boost edges
                        img_aug = exposure.adjust_gamma(img_aug, 0.8)
                    
                    if features['contrast'] < 0.5:
                        # Boost contrast
                        img_aug = exposure.rescale_intensity(img_aug)
                    
                    # Apply augmentation
                    img_aug = augmenter(image=img_aug)
                    
                    # Malignant features
                    
                    # Enlarge nuclei
                    h, w = img_aug.shape[:2]
                    src_points = np.float32([[w*0.1, h*0.1], [w*0.9, h*0.1], [w*0.1, h*0.9], [w*0.9, h*0.9]])
                    dst_points = np.float32([[w*0.1-5, h*0.1-5], [w*0.9+5, h*0.1-5], 
                                            [w*0.1-5, h*0.9+5], [w*0.9+5, h*0.9+5]])
                    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    img_aug = cv2.warpPerspective(img_aug, transform_matrix, (w, h))
                    
                    # Boost nuclei colors
                    b, g, r = cv2.split(img_aug)
                    b = np.clip(b * 1.1, 0, 255).astype(np.uint8)  # Blue = nuclei
                    img_aug = cv2.merge([b, g, r])
                    
                else:  # Other classes
                    # Standard aug
                    img_aug = augmenter(image=img_aug)
                
                # Convert back if needed
                if np.max(img) <= 1.0:
                    img_aug = img_aug.astype(np.float32) / 255.0
                
                # Store synthetics
                synthetic_images.append(img_aug)
                synthetic_labels.append(target_class)
                
                # Create synthetic ID
                synthetic_patient_ids.append(-1000 - i)
                
            except Exception as e:
                print(f"Error creating synthetic sample {i}: {e}")
        
        if len(synthetic_images) == 0:
            print("WARNING: Failed to create any synthetic samples")
            return np.array([]), np.array([]), np.array([])
        
        return np.array(synthetic_images), np.array(synthetic_labels), np.array(synthetic_patient_ids)
    
    def _extract_patient_id(self, img_path):
        # Get patient ID
        # Get filename
        filename = os.path.basename(img_path)
        
        # Try ID patterns
        patterns = [
            r'patient[_-]?(\d+)',
            r'id[_-]?(\d+)',
            r'p(\d+)',
            r'patient(\d+)',
            r'slide(\d+)',
            r'sample[_-]?(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                return int(match.group(1))
        
        # Fallback: use folder
        path_parts = os.path.normpath(img_path).split(os.sep)
        folder_path = os.sep.join(path_parts[:-1])
        
        # Store mapping
        if folder_path not in self.patient_mapping:
            self.patient_mapping[folder_path] = len(self.patient_mapping)
            
        return self.patient_mapping[folder_path]
    
    def create_data_splits(self, images, labels, patient_ids=None, test_size=0.2,
                          val_size=0.1, random_state=42, stratify=True, patient_aware=True):
        # Create train/val/test splits
        if patient_ids is None:
            if hasattr(self, 'patient_ids'):
                patient_ids = self.patient_ids
            else:
                print("WARNING: No patient IDs available. Generating random IDs.")
                patient_ids = np.arange(len(images))
                patient_aware = False  # Can't split by patient
        
        # Check class count
        if len(np.unique(labels)) < 2:
            print("WARNING: Only one class detected. Stratified split not possible.")
            stratify = False
        
        if patient_aware and stratify:
            try:
                # Patient-aware split
                unique_patients = np.unique(patient_ids)
                
                # Map patient to label
                patient_to_label = {}
                for p in unique_patients:
                    p_mask = patient_ids == p
                    if np.sum(p_mask) > 0:
                        # Get majority label
                        p_labels = labels[p_mask]
                        unique_labels, counts = np.unique(p_labels, return_counts=True)
                        patient_to_label[p] = unique_labels[np.argmax(counts)]
                
                # Get patient labels
                patient_labels = np.array([patient_to_label.get(p, 0) for p in unique_patients])
                
                # First split: train+val / test
                train_val_patients, test_patients = train_test_split(
                    unique_patients, 
                    test_size=test_size,
                    random_state=random_state,
                    stratify=patient_labels
                )
                
                # Second split: train / val
                val_ratio = val_size / (1 - test_size)
                
                # Get train_val labels
                train_val_patient_labels = np.array([patient_to_label.get(p, 0) for p in train_val_patients])
                
                train_patients, val_patients = train_test_split(
                    train_val_patients,
                    test_size=val_ratio,
                    random_state=random_state,
                    stratify=train_val_patient_labels
                )
                
                # Create masks
                train_mask = np.isin(patient_ids, train_patients)
                val_mask = np.isin(patient_ids, val_patients)
                test_mask = np.isin(patient_ids, test_patients)
                
                # Apply masks
                X_train, y_train = images[train_mask], labels[train_mask]
                X_val, y_val = images[val_mask], labels[val_mask]
                X_test, y_test = images[test_mask], labels[test_mask]
                
                print("Patient-aware stratified split completed successfully")
                
            except Exception as e:
                print(f"Error in patient-aware stratified split: {e}")
                print("Falling back to regular stratified split")
                
                # Fallback to standard
                patient_aware = False
                stratify = True
        
        # Standard split
        if not patient_aware:
            strat_labels = labels if stratify else None
            
            # First split
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                images, labels,
                test_size=test_size,
                random_state=random_state,
                stratify=strat_labels
            )
            
            # Second split
            strat_train_val = y_train_val if stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size / (1 - test_size),
                random_state=random_state,
                stratify=strat_train_val
            )
        
        # Print stats
        print(f"Train set: {X_train.shape} - Class distribution: {np.bincount(y_train)}")
        print(f"Val set: {X_val.shape} - Class distribution: {np.bincount(y_val)}")
        print(f"Test set: {X_test.shape} - Class distribution: {np.bincount(y_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_kfold_splits(self, images, labels, patient_ids=None, n_splits=5, 
                           random_state=42, patient_aware=True):
        # Create k-fold splits
        if patient_ids is None:
            if hasattr(self, 'patient_ids'):
                patient_ids = self.patient_ids
            else:
                print("WARNING: No patient IDs available. Using regular k-fold.")
                patient_aware = False
        
        if patient_aware:
            try:
                # Patient-aware kfold
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                splits = list(cv.split(images, labels, groups=patient_ids))
                print(f"Created {n_splits} patient-aware stratified folds")
                return splits
            except Exception as e:
                print(f"Error in patient-aware k-fold: {e}")
                print("Falling back to regular stratified k-fold")
                patient_aware = False
        
        # Regular kfold
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(images, labels))
        print(f"Created {n_splits} stratified folds")
        return splits
    
    def apply_mixup(self, images, labels, alpha=0.2):
        # Mix image pairs
        batch_size = len(images)
        
        # Convert to one-hot
        if len(labels.shape) == 1:
            n_classes = max(labels) + 1
            labels_onehot = np.zeros((batch_size, n_classes))
            labels_onehot[np.arange(batch_size), labels] = 1
        else:
            labels_onehot = labels
        
        # Get mix weights
        weights = np.random.beta(alpha, alpha, batch_size)
        weights = np.maximum(weights, 1 - weights)
        weights = weights.reshape(batch_size, 1, 1, 1)
        
        # Get permutation
        indices = np.random.permutation(batch_size)
        
        # Mix images
        mixed_images = weights * images + (1 - weights) * images[indices]
        
        # Mix labels
        weights = weights.reshape(batch_size, 1)
        mixed_labels = weights * labels_onehot + (1 - weights) * labels_onehot[indices]
        
        return mixed_images, mixed_labels
    
    def apply_cutmix(self, images, labels, alpha=1.0):
        # Cut and mix images
        batch_size = len(images)
        
        # Convert to one-hot
        if len(labels.shape) == 1:
            n_classes = max(labels) + 1
            labels_onehot = np.zeros((batch_size, n_classes))
            labels_onehot[np.arange(batch_size), labels] = 1
        else:
            labels_onehot = labels
        
        # Get beta value
        lam = np.random.beta(alpha, alpha)
        
        # Get dimensions
        h, w = images.shape[1:3]
        
        # Get box size
        cut_w = int(w * np.sqrt(1 - lam))
        cut_h = int(h * np.sqrt(1 - lam))
        
        # Get box center
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Get box coords
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2 = min(h, cy + cut_h // 2)
        
        # Get permutation
        indices = np.random.permutation(batch_size)
        
        # Create cutmix images
        mixed_images = images.copy()
        mixed_images[:, y1:y2, x1:x2, :] = images[indices, y1:y2, x1:x2, :]
        
        # Adjust lambda
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (w * h)
        
        # Mix labels
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[indices]
        
        return mixed_images, mixed_labels
    
    def get_augmented_batch(self, images, labels, batch_size=32, augment_type='standard'):
        # Get augmented batch
        # Select indices
        indices = np.random.choice(len(images), batch_size, replace=len(images) < batch_size)
        
        # Get batch
        batch_images = images[indices]
        batch_labels = labels[indices]
        
        # Convert if float
        is_float = batch_images.dtype != np.uint8 and np.max(batch_images) <= 1.0
        if is_float:
            batch_images_aug = (batch_images * 255).astype(np.uint8)
        else:
            batch_images_aug = batch_images.copy()
        
        # Apply augmentation
        augmenter = self.augmenters.get(augment_type, self.augmenters['standard'])
        batch_images_aug = augmenter(images=batch_images_aug)
        
        # Convert back
        if is_float:
            batch_images_aug = batch_images_aug.astype(np.float32) / 255.0
        
        # Apply mixup/cutmix
        if np.random.rand() < 0.5 and augment_type == 'standard':
            if np.random.rand() < 0.5:
                # Apply mixup
                batch_images_aug, batch_labels_aug = self.apply_mixup(batch_images_aug, batch_labels, alpha=0.2)
            else:
                # Apply cutmix
                batch_images_aug, batch_labels_aug = self.apply_cutmix(batch_images_aug, batch_labels, alpha=1.0)
        else:
            # Keep original labels
            if len(labels.shape) == 1:
                n_classes = max(labels) + 1
                batch_labels_aug = np.zeros((batch_size, n_classes))
                batch_labels_aug[np.arange(batch_size), batch_labels] = 1
            else:
                batch_labels_aug = batch_labels
        
        return batch_images_aug, batch_labels_aug
    
    def get_tta_batches(self, images, n_augmentations=5):
        # Test-time augmentation
        # Init storage
        tta_images = []
        
        # Convert if float
        is_float = images.dtype != np.uint8 and np.max(images) <= 1.0
        if is_float:
            images_uint8 = (images * 255).astype(np.uint8)
        else:
            images_uint8 = images.copy()
        
        # Add originals
        tta_images.append(images)
        
        # Get TTA augmenter
        augmenter = self.augmenters['tta']
        
        # Create augmented versions
        for i in range(n_augmentations - 1):
            # Apply augmentation
            aug_images = augmenter(images=images_uint8)
            
            # Convert back
            if is_float:
                aug_images = aug_images.astype(np.float32) / 255.0
            
            tta_images.append(aug_images)
        
        # Stack and return
        return np.array(tta_images)
    
    def visualize_batch(self, images, labels, predictions=None, class_names=None, n_samples=10, figsize=(15, 10)):
        # Visualize batch
        # Limit samples
        n_samples = min(n_samples, len(images))
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 1, figsize=figsize)
        if n_samples == 1:
            axes = [axes]
        
        # Default class names
        if class_names is None:
            class_names = [f"Class {i}" for i in range(max(labels) + 1)]
        
        # Get label indices
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            label_indices = np.argmax(labels, axis=1)
        else:
            label_indices = labels
        
        # Process predictions
        pred_indices = None
        pred_probs = None
        if predictions is not None:
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Probabilities
                pred_indices = np.argmax(predictions, axis=1)
                pred_probs = predictions
            else:
                # Class indices
                pred_indices = predictions
        
        # Show each sample
        for i in range(n_samples):
            ax = axes[i]
            
            # Convert for display
            img = images[i]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            # Show image
            ax.imshow(img)
            
            # Create title
            title = f"True: {class_names[label_indices[i]]}"
            
            # Add prediction
            if pred_indices is not None:
                title += f" | Pred: {class_names[pred_indices[i]]}"
                
                # Add probability
                if pred_probs is not None:
                    title += f" ({pred_probs[i][pred_indices[i]]:.2f})"
                    
                    # Add correct indicator
                    if pred_indices[i] == label_indices[i]:
                        title += " ✓"
                    else:
                        title += " ✗"
            
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_augmentations(self, image, n_augmentations=5, figsize=(15, 10)):
        # Show augmentations
        # Create figure
        fig, axes = plt.subplots(1, n_augmentations + 1, figsize=figsize)
        
        # Convert if float
        is_float = image.dtype != np.uint8 and np.max(image) <= 1.0
        if is_float:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.copy()
        
        # Show original
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Get augmenter
        augmenter = self.augmenters['standard']
        
        # Show augmented
        for i in range(n_augmentations):
            # Apply augmentation
            aug_image = augmenter(image=image_uint8.copy())
            
            # Convert back
            if is_float:
                aug_image = aug_image.astype(np.float32) / 255.0
            
            # Display
            axes[i + 1].imshow(aug_image)
            axes[i + 1].set_title(f"Augmentation {i + 1}")
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()