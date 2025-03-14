# Leukemia Classifier Version: Summary

## V1:

### Dataset & Setup
- **Data**: 3,242 blood smear images across 4 classes (Benign, Pre-B, Pro-B, early Pre-B)
- **Split**: 80/20 train/test (2,593/649 images)
- **Size**: 224×224 pixels RGB

### Model
- **Architecture**: ResNet50 backbone (frozen) + custom classification head
- **Training**: 
  - Adam optimizer (1e-4 initial LR)
  - Class weighting to handle imbalance
  - Data augmentation (rotation, shifts, zoom, flip)
  - Early stopping at 39 epochs

### Results
- **Test Accuracy**: 53.93%
- **Class Performance**:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Benign | 0.28 | 0.17 | 0.21 |
| Pre-B | 0.47 | 0.82 | 0.60 |
| Pro-B | 0.78 | 0.87 | 0.82 |
| early Pre-B | 0.47 | 0.19 | 0.27 |

### Key Observations
- Pro-B classification exceptionally strong (F1: 0.82)
- Significant weakness in Benign and early Pre-B detection
- Model successfully trains without computational issues
- 54% accuracy significantly better than random (25%)

## V2:

### Dataset & Setup
- **Data**: Currently only 1024 Benign class images (other classes missing)
- **Split**: 80/20 train/test (819/205 images)
- **Size**: 384×384 pixels RGB (increased from V1 and V2)
- **Approach**: Synthetic data generation for missing classes

### Model
- **Architecture**: DenseNet201 backbone + enhanced attention mechanism
- **Key Features**:
  - CBAM-inspired channel and spatial attention blocks
  - Monte Carlo Dropout for uncertainty estimation
  - Focal Loss for handling class imbalance
  - Three-phase gradual unfreezing strategy

### Training
- **Optimizer**: AdamW with weight decay (1e-4 initial LR, 1e-5 weight decay)
- **Augmentation**: Advanced pipeline creating 200 synthetic samples per missing class
- **Training Strategy**: Three-phase training with gradual unfreezing
  - Phase 1: Frozen backbone
  - Phase 2: Unfreeze last 20% of backbone layers
  - Phase 3: Unfreeze entire model

### Results
- **Best Val Loss**: 1.34622 (Epoch 1)
- **Training Progress**:
  - Rapid overfitting after epoch 1
  - Validation loss increased to 3.989 by epoch 6
  - Validation accuracy dropped to near 0%
  - Validation precision/recall dropped to 0.0

### Key Observations
- Model architecture too complex for current dataset
- Synthetic data approach ineffective for class balance
- Gradual unfreezing strategy causing overfitting with synthetic data
- Higher resolution (384×384) increasing computational cost without benefit
- Attention mechanism potentially overparameterizing the model

### Recommendations for Future Iterations
1. **Simplify Architecture**: Use lighter backbone (EfficientNetB0/ResNet18)
2. **Binary Classification**: Focus on Benign vs. Non-Benign until all class data is available
3. **Regularization**: Increase dropout rate and L2 regularization
4. **Reduced Learning Rate**: Start with 5e-5 instead of 1e-4
5. **Simpler Augmentation**: Less aggressive synthetic data generation
6. **Data Collection**: Focus on obtaining real samples for missing classes

## V3:

### Dataset & Setup
- **Split**: 5-fold cross-validation with patient-aware grouping option
- **Size**: 256×256 pixels RGB (reduced from V2's 384×384)
- **Approach**: 100 synthetic samples generated for each missing class

### Model
- **Architecture**: ResNet50 backbone + CBAM-inspired attention mechanism
- **Key Features**:
  - Monte Carlo Dropout (rate=0.6) for uncertainty estimation
  - Conservative 3-phase training approach
  - Class weighting with beta smoothing for imbalance

### Training
- **Optimizer**: AdamW with reduced learning rate (5e-5) and weight decay (2e-5)
- **Augmentation**: More conservative augmentation strategy
- **Training Strategy**: Early stopping with patience=20
  - Training stopped at epoch 63 with no improvement from epoch 43

### Results
- **Test Accuracy**: 40.49%
- **Class Performance**:

| Class | Precision | Recall | F1-Score | Specificity |
|-------|-----------|--------|----------|-------------|
| Benign | 1.0000 | 0.1902 | 0.3197 | 0.0000 |
| Pre-B | 0.0000 | 0.0000 | 0.0000 | 0.7073 |
| Pro-B | 0.0000 | 0.0000 | 0.0000 | 0.7659 |
| early Pre-B | 0.0000 | 0.0000 | 0.0000 | 0.7171 |

### Key Observations
- Model shows improved stability compared to V2 but still struggles with synthetic classes
- Only identifies Benign samples with perfect precision but poor recall (19%)
- Synthetic samples not sufficiently realistic for classifier to learn meaningful features
- Reduced image size and learning rate improved training stability
- Positive: Training process completed without crashing or extreme overfitting

### Recommendations for Future Iterations
1. **Real Data Collection**: Critical to obtain actual samples for ALL subtypes
2. **Binary Classification**: Implement Benign vs. ALL binary classifier as interim solution
3. **External Pre-training**: Consider models pre-trained on medical imaging datasets
4. **Active Learning**: Implement uncertainty-based active learning for efficient labeling
5. **Advanced Synthetic Approaches**: Explore GAN-based synthetic sample generation
6. **Ensemble Methods**: Create ensemble of binary classifiers instead of multi-class approach

## V4:

### Dataset & Setup
- **Data**: 1,024 benign samples + 100 synthetic malignant samples
- **Split**: 80/20 train/test (819/205 benign, 80/20 synthetic malignant)
- **Size**: 224×224 pixels RGB
- **Approach**: Binary classification (Benign vs. ALL) with synthetic samples for missing class

### Model
- **Architecture**: EfficientNetB0 backbone (lighter than previous versions)
- **Key Features**:
  - Monte Carlo Dropout for uncertainty estimation
  - Simplified 2-phase training with gradual unfreezing
  - Using legacy optimizers for M1/M2 Mac compatibility

### Training
- **Optimizer**: Legacy Adam with reduced learning rate (5e-5) and gradient clipping
- **Augmentation**: Conservative synthetic data generation for malignant samples
- **Training Strategy**: 5 epochs initial + fine-tuning with early stopping
  - Best validation loss at epoch 1, showing signs of overfitting thereafter

### Results
- **Test Accuracy**: 84.00%
- **Performance Metrics**:

| Metric | Value |
|--------|-------|
| Precision | 0.1000 |
| Recall | 0.1000 |
| F1 Score | 0.1000 |
| Specificity | 0.9122 |
| AUC | 0.3898 |

- **Confusion Matrix**:
  - True Negatives: 187
  - False Positives: 18
  - False Negatives: 18
  - True Positives: 2

### Key Observations
- High accuracy (84%) is misleading due to class imbalance
- Very poor detection of malignant samples (only 2 true positives)
- AUC below 0.5 indicates model performance worse than random for prediction ranking
- Model successfully completes training without technical errors
- While code structure improvements were made, the fundamental data imbalance remains problematic

### Recommendations for Next Steps
1. **Obtain Real Malignant Samples**: Synthetic data approach has significant limitations
2. **Transfer Learning**: Consider using models pre-trained on similar cell images
3. **Class Weighting**: Further increase class weight for minority class
4. **Feature Engineering**: Extract cell morphology features prior to classification
5. **External Validation**: Test performance on external dataset if available
6. **Improve Synthetic Data Quality**: Explore GANs for generating more realistic samples


## V5:

### Dataset & Setup
- **Data**: 1,433 blood smear images (1,024 Benign, 409 Malignant)
- **Split**: Standard train/validation/test split
- **Size**: 224×224 pixels RGB
- **Approach**: Binary classification (Benign vs. ALL)

### Model
- **Architecture**: EfficientNetB0 backbone with advanced attention mechanism
- **Key Features**:
  - Convolutional Block Attention Module (CBAM)
  - Focal Loss for handling class imbalance
  - Monte Carlo Dropout for uncertainty estimation
  - Gradual fine-tuning strategy

### Training
- **Optimizer**: RectifiedAdam with learning rate scheduling
- **Augmentation**: Advanced data augmentation pipeline
- **Training Strategy**: Two-phase training with initial frozen backbone and gradual unfreezing

### Results
- **Test Accuracy**: 81.02%
- **Performance Metrics**:

| Metric | Value |
|--------|-------|
| Precision | 0.6424 |
| Recall | 0.7555 |
| F1 Score | 0.6944 |
| Specificity | 0.8320 |
| AUC | 0.8515 |
| Balanced Accuracy | 0.7938 |
| Optimal Threshold | 0.3619 |
| Brier Score | 0.2065 |

- **Confusion Matrix**:
  - True Negatives: 852
  - False Positives: 172
  - False Negatives: 100
  - True Positives: 309

### Key Observations
- Significant improvement over previous versions
- Strong ability to distinguish between Benign and Malignant classes
- High specificity (83.20%) indicates low false positive rate
- AUC of 0.8515 shows good predictive ranking capability
- Balanced accuracy of 0.7938 suggests robust performance across classes

### Recommendations for Future Iterations
1. Collect more malignant samples to further balance the dataset
2. Explore additional feature extraction techniques
3. Investigate more advanced attention mechanisms
4. Consider multi-modal approaches (combining morphological features)
5. Perform extensive external validation