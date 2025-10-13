# 🐶🐱 Dog vs Cat Classification using Support Vector Machine (SVM)

A comprehensive implementation of **SVM-based image classification** with detailed comparison of two feature extraction techniques: **HOG** and **PCA**.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green)](https://opencv.org/)

---

## 📋 Table of Contents

- [About](#-about)
- [Dataset](#-dataset)
- [Feature Extraction Methods](#-feature-extraction-methods)
  - [Method 1: HOG](#method-1-hog-histogram-of-oriented-gradients)
  - [Method 2: PCA](#method-2-pca-principal-component-analysis)
- [SVM Models](#-svm-models)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Key Takeaways](#-key-takeaways)
- [References](#-references)

---

## 🎯 About

This project implements **binary image classification** to distinguish between **Dogs** and **Cats** using **Support Vector Machines (SVM)**. The implementation explores:

- ✅ Two different feature extraction techniques (HOG and PCA)
- ✅ Three SVM kernel types (Linear, RBF, Polynomial)
- ✅ Six total model combinations
- ✅ Comprehensive evaluation with 8+ metrics
- ✅ Visual comparisons and analysis
- ✅ Production-ready code with error handling

**Key Question:** Which feature extraction method works better for image classification with SVM?

---

## 📁 Dataset

**Microsoft Dogs vs Cats Dataset**

- **Source**: Microsoft Research
- **Total Images**: ~25,000 JPG images
- **Classes**: 2 (Binary Classification)
  - **Cat** (Label 0): ~12,500 images
  - **Dog** (Label 1): ~12,500 images
- **Format**: Variable-sized RGB images
- **Balance**: Perfectly balanced dataset

### Preprocessing Steps:

1. **Resize**: All images resized to 128×128 pixels
2. **Grayscale Conversion**: RGB → Grayscale (reduces from 3 channels to 1)
3. **Normalization**: Pixel values scaled to [0, 1]
4. **Error Handling**: Corrupted images automatically skipped
5. **Train-Test Split**: 80% training, 20% testing (stratified)

---

## 🎨 Feature Extraction Methods

Feature extraction is **critical** for image classification with traditional ML algorithms like SVM. Raw pixel values (128×128 = 16,384 dimensions) are too high-dimensional and don't capture meaningful patterns.

### Why Feature Extraction?

| Problem | Solution |
|---------|----------|
| **Too many dimensions** | Reduce to meaningful features |
| **Curse of dimensionality** | Extract discriminative information |
| **No semantic understanding** | Capture edges, shapes, patterns |
| **Computational cost** | Faster training with fewer features |
| **Noise in raw pixels** | Focus on important characteristics |

---

## Method 1: HOG (Histogram of Oriented Gradients)

### 📖 What is HOG?

**HOG** is a feature descriptor that captures the **distribution of edge directions** in an image. It was originally developed for pedestrian detection and has become a standard technique in computer vision.

### 🔍 How HOG Works: Step-by-Step

#### Step 1: Compute Gradients

Calculate horizontal and vertical gradients at each pixel to detect edges:

```
Horizontal Gradient (Gx):     Vertical Gradient (Gy):
[-1, 0, +1]                    [-1]
                               [ 0]
                               [+1]

Gradient Magnitude: √(Gx² + Gy²)
Gradient Direction: arctan(Gy / Gx)
```

**Example:**
```
Original Image Patch:        Gradient Magnitude:      Gradient Direction:
10  20  30                   14  14  14               →  →  →
15  25  35                   14  14  14               →  →  →
20  30  40                   14  14  14               →  →  →
```

The gradients show edge strength and direction at each pixel.

---

#### Step 2: Divide into Cells

Split the image into small **cells** (typically 8×8 pixels each):

```
128×128 Image → 16×16 Grid of 8×8 Cells
```

**Why cells?**
- Capture local features while maintaining spatial structure
- More robust than individual pixels
- Reduce computational complexity

---

#### Step 3: Create Histograms per Cell

For each cell, create a histogram of gradient directions:

- **Bins**: Usually 9 orientation bins (0°, 20°, 40°, ..., 160°)
- **Vote**: Each pixel votes for its gradient direction
- **Weight**: Votes weighted by gradient magnitude

**Visualization:**

```
Cell (8×8 pixels) → Histogram (9 bins)

        90°
         |
         |
180° ----+---- 0°
         |
         |
        270°

Histogram: [h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉]
           ▂▄█▆▃▁▂▅▇  (Example distribution)
```

**Example:**
If a cell has many horizontal edges, the histogram will have high values in the 0° and 180° bins.

---

#### Step 4: Normalize over Blocks

Group cells into larger **blocks** (typically 2×2 cells) and normalize:

```
Block (2×2 cells) = 4 cells × 9 bins = 36 values
Normalization: v' = v / √(‖v‖² + ε²)
```

**Why normalization?**
- **Robust to lighting changes**: Bright vs. dark images
- **Contrast invariance**: Same edges regardless of intensity
- **Reduces sensor noise**: More stable features

---

#### Step 5: Concatenate into Feature Vector

Collect all block histograms into a single feature vector:

```
For 128×128 image:
- 15×15 blocks (with overlap)
- Each block: 36 features
- Total: 15×15×36 = 8,100 features
```

**Feature vector reduction:**
- Original: 128×128 = 16,384 pixels
- HOG: ~8,100 features
- **Reduction: ~50%** while capturing edge information

---

### 🎯 HOG Visual Example

```
Original Image:              Gradient Image:           HOG Visualization:
   🐱                        ████░░░░████              ╱╲╱╲╱╲
  /oo\                       ██░░▒▒░░██                │││││
  \__/                       ░░████████░░              ╲╱╲╱╲
                             ████░░░░████              ╱╲╱╲╱

The HOG descriptor captures edge orientations:
- Vertical edges (ears, sides)
- Horizontal edges (eyes, mouth)
- Diagonal edges (face shape)
```

---

### 💡 HOG Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| **orientations** | 9 | Number of gradient direction bins |
| **pixels_per_cell** | (8, 8) | Size of each cell in pixels |
| **cells_per_block** | (2, 2) | Cells per normalization block |
| **visualize** | True/False | Generate HOG visualization |
| **feature_vector** | True | Flatten output into 1D vector |

---

### ✅ HOG Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Illumination Invariant** | Block normalization handles lighting changes |
| **Captures Shape** | Edge orientations describe object shapes |
| **Geometric Transformation** | Somewhat robust to small rotations/translations |
| **Proven Track Record** | Used in pedestrian detection, face recognition |
| **Domain-Specific** | Designed specifically for computer vision |
| **Interpretable** | Can visualize what features are captured |

---

### ❌ HOG Limitations

| Limitation | Impact |
|------------|--------|
| **Fixed Scale** | Doesn't handle objects at different scales well |
| **Computational Cost** | Slower than PCA for large images |
| **Sensitive to Occlusion** | Partial objects may not be recognized |
| **Hand-Crafted** | Requires domain knowledge to design |
| **More Features** | ~8,100 features vs fewer with PCA |

---

## Method 2: PCA (Principal Component Analysis)

### 📖 What is PCA?

**PCA** is a statistical technique that transforms high-dimensional data into a lower-dimensional space by finding directions (**principal components**) that capture maximum variance in the data.

### 🔍 How PCA Works: Step-by-Step

#### Step 1: Flatten Images

Convert 2D image into 1D vector:

```
128×128 Image → 16,384-dimensional vector

Original:              Flattened:
[[ 10  20  30  ... ]   [10, 20, 30, 15, 25, 35, ...]
 [ 15  25  35  ... ]   (16,384 values)
 [ 20  30  40  ... ]
 ...              ]
```

---

#### Step 2: Standardize the Data

Center data by subtracting mean:

```
X' = X - mean(X)

Why? PCA finds directions of maximum variance.
Without centering, variance includes the mean shift.
```

**Example:**

```
Original Pixel Values:    Centered Values:
[100, 150, 200]          [-50, 0, +50]

Mean = 150
Centered = Original - 150
```

---

#### Step 3: Compute Covariance Matrix

Calculate relationships between all pixel pairs:

```
Covariance Matrix C = (1/n) × X'ᵀ × X'

Size: 16,384 × 16,384 (huge!)

C[i,j] = covariance between pixel i and pixel j
```

**What does covariance tell us?**
- **Positive**: Pixels vary together (both bright or both dark)
- **Negative**: Pixels vary oppositely (one bright, other dark)
- **Zero**: No relationship

**Example:**
```
Cat images: Eyes are usually dark
→ Eye pixels have high covariance with each other
→ PCA captures this pattern
```

---

#### Step 4: Compute Eigenvectors and Eigenvalues

Find principal components (eigenvectors) and their importance (eigenvalues):

```
Solve: C × v = λ × v

Where:
- v = eigenvector (principal component direction)
- λ = eigenvalue (variance explained by that direction)

Sort eigenvectors by eigenvalues (largest first)
```

**Intuition:**

```
Original Data:           First Principal Component:
    •  •                     ↗ ↗ ↗
  •  •  •                   ↗ ↗ ↗
    •  •                   ↗ ↗ ↗

PC1 = Direction of maximum variance
PC2 = Perpendicular direction (2nd most variance)
...
```

**Visual Analogy:**

Imagine a football field with scattered footballs:
- **PC1**: Length direction (most variation)
- **PC2**: Width direction (2nd most variation)
- **PC3**: Height direction (least variation)

---

#### Step 5: Select Top Components

Choose principal components that explain desired variance (e.g., 95%):

```
Explained Variance Ratio:
PC1: 15.2% ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
PC2:  8.3% ▓▓▓▓▓▓▓▓
PC3:  5.1% ▓▓▓▓▓
...
PC100: 0.2% ▓

Cumulative: PC1-200 explain 95% of variance
```

**Example:**
- Original: 16,384 dimensions
- Keep: 200 components
- **Reduction: 98.8%**
- Preserve: 95% of information

---

#### Step 6: Project Data

Transform original data onto principal components:

```
X_reduced = X' × V

Where:
- X' = Centered data (n × 16,384)
- V = Selected eigenvectors (16,384 × 200)
- X_reduced = Transformed data (n × 200)
```

**Visualization:**

```
Original Space (16,384D):     PCA Space (200D):
        •                            •
    •   •   •                    •   •   •
  •   •   •   •        →           •   •
    •   •   •                    •   •   •
        •                            •

Data projected onto principal directions
```

---

### 🎯 PCA Visual Example

#### Eigenfaces (Principal Components)

When PCA is applied to images, the principal components look like "ghost faces":

```
PC1 (15% var):     PC2 (8% var):      PC3 (5% var):
   ███░░░███          ░░░███░░░         ███░░░███
   ██░▒▒░██           ███░▒░███         ░░░▒▒▒░░░
   ░░████░░           ░░█████░░         ███████░░
   ████░████          ████░████         ░░░███░░░

Each component captures different patterns:
- PC1: Average face structure
- PC2: Lighting variation
- PC3: Expression differences
```

#### Variance Explained

```
Cumulative Variance:
100%|
    |                         ____________
90% |                    ___/
    |               ___/
70% |          ___/
    |      __/
50% |   __/
    | /
 0% +────────────────────────────────────
    0    50   100   150   200   250   300
         Number of Components

95% threshold reached at ~200 components
```

---

### 💡 PCA Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| **n_components** | 0.95 | Keep components explaining 95% variance |
| **whiten** | False | Keep original scale (True for SVM may help) |
| **random_state** | 42 | For reproducibility |

---

### ✅ PCA Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Massive Dimensionality Reduction** | 16,384 → ~200 features (98.8% reduction) |
| **Removes Redundancy** | Nearby pixels are highly correlated |
| **Data-Driven** | Learns from actual data patterns |
| **Fast Computation** | Once computed, transformation is matrix multiplication |
| **Noise Reduction** | Low-variance components often contain noise |
| **General Purpose** | Works for any high-dimensional data |
| **Fewer Features** | Much faster SVM training |

---

### ❌ PCA Limitations

| Limitation | Impact |
|------------|--------|
| **Linear Only** | Can't capture non-linear patterns |
| **Global Method** | Doesn't preserve local structure |
| **No Domain Knowledge** | Doesn't know about edges, shapes, etc. |
| **Variance ≠ Discriminative** | High variance may not help classification |
| **Not Interpretable** | Hard to understand what each component means |
| **Sensitive to Scale** | Must standardize data first |

---

## 🆚 HOG vs PCA: Detailed Comparison

### Feature Extraction Philosophy

| Aspect | HOG | PCA |
|--------|-----|-----|
| **Design** | Hand-crafted, domain-specific | Data-driven, general-purpose |
| **Information** | Edge orientations & magnitudes | Statistical variance patterns |
| **Prior Knowledge** | Uses computer vision principles | Learns from data |
| **Feature Type** | Histogram of gradients | Linear combinations of pixels |
| **Spatial Awareness** | Preserves local structure | Global transformation |

---

### Mathematical Comparison

| Aspect | HOG | PCA |
|--------|-----|-----|
| **Input** | Gradients (edge information) | Raw pixel intensities |
| **Transformation** | Gradient → Histogram → Normalize | Pixel → Center → Project |
| **Output Size** | ~8,100 features | ~200 features (95% variance) |
| **Computation** | Gradient filters + histograms | Eigenvalue decomposition |
| **Complexity** | O(n × m × k) pixels × cells × bins | O(d³) features cubed |

---

### Performance Characteristics

| Aspect | HOG | PCA |
|--------|-----|-----|
| **Training Time** | Slower (more features) | Faster (fewer features) |
| **Prediction Time** | Slower (more features) | Faster (fewer features) |
| **Memory Usage** | Higher (~8,100 × n samples) | Lower (~200 × n samples) |
| **Robustness** | Good to lighting, weak to rotation | Sensitive to transformations |
| **Interpretability** | Can visualize edge patterns | Hard to interpret components |

---

### When to Use Which?

#### ✅ Use HOG When:

1. **Computer Vision Task**: Object detection, recognition
2. **Edge Information Important**: Shapes, boundaries matter
3. **Domain Knowledge Available**: Know what features to look for
4. **Interpretability Needed**: Want to understand what's captured
5. **Accuracy Priority**: Can afford computation for better results
6. **Small Rotation/Translation**: Objects mostly aligned

**Example Use Cases:**
- Pedestrian detection
- Face recognition
- Traffic sign recognition
- Character recognition (OCR)

---

#### ✅ Use PCA When:

1. **General Data**: Not specific to images
2. **Dimensionality Reduction Priority**: Need fewer features
3. **Speed Critical**: Fast training/prediction required
4. **Data-Driven Approach**: Let data determine features
5. **Memory Constrained**: Limited storage
6. **High-Dimensional Input**: Need massive reduction

**Example Use Cases:**
- High-dimensional datasets
- Pre-processing for neural networks
- Compression
- Visualization (reduce to 2D/3D)
- Genomics, finance data

---

### Hybrid Approach

**Best of Both Worlds:**

```python
# 1. Extract HOG features (domain knowledge)
hog_features = extract_hog(images)

# 2. Apply PCA on HOG (dimensionality reduction)
pca_hog_features = pca.fit_transform(hog_features)

# 3. Train SVM on reduced features
svm.fit(pca_hog_features, labels)
```

**Advantages:**
- HOG captures meaningful edges
- PCA removes HOG feature redundancy
- Best accuracy + speed balance

---

## 🤖 SVM Models

### Support Vector Machine Overview

**SVM** finds the optimal hyperplane that **maximally separates** classes with the **largest margin**.

```
Binary Classification:

         Margin
    |----------------|
    |                |
    |  •  •  •       |      • = Class 0 (Cat)
    | •  •  •        |      ○ = Class 1 (Dog)
    |                |      ■ = Support Vectors
========■============■======== Optimal Hyperplane
    |                |
    |        ○  ○  ○ |
    |       ○  ○  ○  |
    |                |
```

**Key Concepts:**

1. **Support Vectors**: Data points closest to hyperplane
2. **Margin**: Distance between hyperplane and nearest points
3. **Kernel Trick**: Project data to higher dimensions for non-linear separation

---

### Three SVM Kernels

#### 1. Linear SVM

**Decision Function:**
```
f(x) = sign(w·x + b)
```

**When to use:**
- Data is linearly separable
- Want fast training/prediction
- Need interpretable model
- High-dimensional data (text, HOG)

**Pros:**
- ✅ Fastest training and prediction
- ✅ Works well with high dimensions
- ✅ Good baseline model
- ✅ Least prone to overfitting

**Cons:**
- ❌ Only captures linear relationships
- ❌ May underfit complex data

---

#### 2. RBF (Radial Basis Function) SVM

**Kernel Function:**
```
K(x, x') = exp(-γ × ‖x - x'‖²)

Where γ controls the influence of single training example
```

**Intuition:**
- Measures similarity between points
- Close points → high similarity (≈1)
- Far points → low similarity (≈0)

**When to use:**
- Non-linear relationships exist
- No prior knowledge of data structure
- Want good default kernel
- Can afford computational cost

**Pros:**
- ✅ Most popular kernel (good default)
- ✅ Handles non-linear patterns well
- ✅ Works for various problems
- ✅ Flexible decision boundaries

**Cons:**
- ❌ Slower than linear
- ❌ More hyperparameters (C, γ)
- ❌ Can overfit with wrong parameters

---

#### 3. Polynomial SVM

**Kernel Function:**
```
K(x, x') = (γ × x·x' + r)^d

Where:
- d = degree of polynomial (default: 3)
- γ = kernel coefficient
- r = independent term
```

**When to use:**
- Polynomial relationships suspected
- Image processing (pixel interactions)
- Limited number of features
- Want specific degree of interaction

**Pros:**
- ✅ Captures polynomial interactions
- ✅ Good for image data
- ✅ Flexible with degree parameter

**Cons:**
- ❌ Can be very slow
- ❌ Prone to overfitting
- ❌ Sensitive to degree choice

---

### SVM Parameters

#### Key Hyperparameters:

| Parameter | Description | Impact |
|-----------|-------------|--------|
| **C** | Regularization strength | High C = less regularization (complex model) |
|  | (default: 1.0) | Low C = more regularization (simple model) |
| **kernel** | Kernel type | linear, rbf, poly, sigmoid |
| **gamma** | Kernel coefficient | High γ = narrow influence (overfitting) |
|  | (for rbf, poly) | Low γ = wide influence (underfitting) |
| **degree** | Polynomial degree | Higher = more complex (poly only) |
| **probability** | Enable probability estimates | True = can get prediction probabilities |

---

## 📊 Evaluation Metrics

We use **8 comprehensive metrics** to evaluate models:

### 1. Accuracy

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretation:**
- Overall correctness
- Good for balanced datasets
- Can be misleading for imbalanced data

**Range:** [0, 1] (higher is better)

---

### 2. Balanced Accuracy

**Formula:**
```
Balanced Accuracy = (Sensitivity + Specificity) / 2
                  = (TPR + TNR) / 2
```

**Interpretation:**
- Average of recall for each class
- Better for imbalanced datasets
- Treats all classes equally

**Range:** [0, 1] (higher is better)

---

### 3. Precision

**Formula:**
```
Precision = TP / (TP + FP)
```

**Interpretation:**
- Of predicted positive, how many are actually positive?
- "How precise are positive predictions?"
- Important when false positives are costly

**Example:** Medical diagnosis (don't want false alarms)

**Range:** [0, 1] (higher is better)

---

### 4. Recall (Sensitivity, True Positive Rate)

**Formula:**
```
Recall = TP / (TP + FN)
```

**Interpretation:**
- Of actual positive, how many did we predict?
- "How many positives did we catch?"
- Important when false negatives are costly

**Example:** Disease detection (don't want to miss cases)

**Range:** [0, 1] (higher is better)

---

### 5. F1-Score

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation:**
- Harmonic mean of precision and recall
- Single metric balancing both
- Good when classes are imbalanced

**Why harmonic mean?** Penalizes extreme values (low precision OR low recall)

**Range:** [0, 1] (higher is better)

---

### 6. ROC-AUC (Area Under ROC Curve)

**ROC Curve:** True Positive Rate vs False Positive Rate

```
TPR |     /─────
    |    /
    |   /
    |  /
    | /
    |/
    └──────────── FPR

AUC = Area under this curve
```

**Interpretation:**
- Probability that model ranks random positive higher than random negative
- Model's ability to distinguish between classes
- Independent of threshold

**Values:**
- 1.0 = Perfect classifier
- 0.9-1.0 = Excellent
- 0.8-0.9 = Good
- 0.7-0.8 = Fair
- 0.5-0.7 = Poor
- 0.5 = Random guessing

**Range:** [0, 1] (higher is better)

---

### 7. Matthews Correlation Coefficient (MCC)

**Formula:**
```
MCC = (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Interpretation:**
- Correlation between predictions and true labels
- Takes all confusion matrix elements into account
- Reliable even for imbalanced datasets
- **Best single metric for binary classification**

**Values:**
- +1 = Perfect prediction
-  0 = Random prediction
- -1 = Complete disagreement

**Range:** [-1, +1] (higher is better)

---

### 8. Cohen's Kappa

**Formula:**
```
κ = (p₀ - pₑ) / (1 - pₑ)

Where:
- p₀ = Observed agreement (accuracy)
- pₑ = Expected agreement by chance
```

**Interpretation:**
- Agreement between predictions and truth, accounting for chance
- How much better than random guessing?

**Values:**
- < 0.00 = Less than chance
- 0.00-0.20 = Slight agreement
- 0.21-0.40 = Fair agreement
- 0.41-0.60 = Moderate agreement
- 0.61-0.80 = Substantial agreement
- 0.81-1.00 = Almost perfect agreement

**Range:** [-1, +1] (higher is better)

---

### Confusion Matrix

```
                Predicted
              Cat    Dog
Actual  Cat  [ TN  |  FP ]
        Dog  [ FN  |  TP ]

TN = True Negative  (Correctly predicted Cat)
FP = False Positive (Predicted Dog, was Cat)
FN = False Negative (Predicted Cat, was Dog)
TP = True Positive  (Correctly predicted Dog)
```

**From confusion matrix, we derive:**
- Accuracy = (TP + TN) / Total
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- Specificity = TN / (TN + FP)

---

## 📂 Project Structure

```
SVM_ImageClassification/
│
├── SVM_DogVsCat.ipynb          # Main Jupyter notebook with complete implementation
├── README.md                   # This file - comprehensive documentation
│
├── Dataset (not included):
│   └── ../../dataset/PetImages/
│       ├── Cat/                # ~12,500 cat images
│       │   ├── 0.jpg
│       │   ├── 1.jpg
│       │   └── ...
│       └── Dog/                # ~12,500 dog images
│           ├── 0.jpg
│           ├── 1.jpg
│           └── ...
│
└── Outputs (generated during execution):
    ├── Models (optional):
    │   ├── svm_linear_hog.pkl
    │   ├── svm_rbf_hog.pkl
    │   └── ...
    └── Visualizations (inline in notebook):
        ├── Sample images
        ├── Preprocessed images
        ├── HOG visualizations
        ├── PCA variance plots
        ├── Confusion matrices
        ├── ROC curves
        └── Performance comparisons
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd MachineLearning/Implementation/SVM_ImageClassification
```

### Step 2: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python scikit-image Pillow
```

**Or use requirements.txt:**

```bash
pip install -r ../../requirements.txt
```

### Step 3: Download Dataset

Download the Microsoft Dogs vs Cats dataset and place it in:
```
../../dataset/PetImages/
```

Structure should be:
```
dataset/
└── PetImages/
    ├── Cat/
    └── Dog/
```

---

## 💻 Usage

### Running the Notebook

1. **Start Jupyter:**
   ```bash
   jupyter notebook SVM_DogVsCat.ipynb
   ```

2. **Run cells sequentially:**
   - Execute each cell in order
   - Wait for long-running cells (feature extraction, training)
   - Review outputs and visualizations

### Expected Runtime

| Operation | Time (approx.) |
|-----------|----------------|
| Loading images | 5-10 minutes |
| HOG extraction | 10-15 minutes |
| PCA computation | 3-5 minutes |
| SVM training (per model) | 5-20 minutes |
| **Total** | **1-2 hours** |

*Times vary based on hardware and dataset size*

### Configuration Options

**Modify these in the notebook:**

```python
# Image size (smaller = faster, less accurate)
IMG_SIZE = 128  # Try 64, 128, 256

# Limit dataset for testing (None = all images)
MAX_IMAGES_PER_CLASS = None  # Try 1000, 5000 for quick tests

# HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# PCA variance to keep
n_components = 0.95  # 95% variance
```

---

## 📈 Results

### Performance Summary

**Expected Results** (will vary based on your run):

#### HOG Features:

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Linear SVM + HOG | ~73% | ~0.73 | ~0.80 |
| RBF SVM + HOG | ~75% | ~0.75 | ~0.82 |
| Poly SVM + HOG | ~74% | ~0.74 | ~0.81 |

#### PCA Features:

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Linear SVM + PCA | ~68% | ~0.68 | ~0.74 |
| RBF SVM + PCA | ~70% | ~0.70 | ~0.77 |
| Poly SVM + PCA | ~69% | ~0.69 | ~0.76 |

### Key Findings

1. **HOG outperforms PCA** by ~5% accuracy
2. **RBF kernel** performs best for both feature types
3. **Linear SVM** surprisingly competitive, much faster
4. **Polynomial kernel** slowest with marginal gains

---

## 💡 Key Takeaways

### 1. Feature Extraction is Critical

- Raw pixels don't work well for SVM
- HOG captures domain-specific information (edges)
- PCA provides general dimensionality reduction
- **Trade-off**: Accuracy (HOG) vs Speed (PCA)

### 2. HOG vs PCA for Images

**HOG wins for accuracy:**
- Designed for computer vision
- Captures meaningful edge patterns
- More robust to lighting

**PCA wins for speed:**
- Fewer features (200 vs 8,100)
- Faster training and prediction
- Lower memory usage

### 3. Kernel Selection Matters

- **Start with RBF** (good default)
- **Try Linear** (fast baseline)
- **Polynomial** rarely worth the cost

### 4. SVM vs Deep Learning

**SVM Advantages:**
- Works with smaller datasets
- Faster to train
- More interpretable
- No GPU required

**Deep Learning Advantages:**
- Better accuracy on large datasets
- Learns features automatically
- Handles complex patterns
- State-of-the-art results

**When to use SVM:**
- < 10,000 images
- Limited computational resources
- Need interpretability
- Quick prototyping

---

## 🔄 Future Improvements

### 1. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
```

### 2. Feature Engineering

- **Combine HOG + PCA**: Apply PCA to HOG features
- **Color Histograms**: Add color information
- **SIFT/SURF**: Other feature descriptors
- **Deep Features**: CNN features from pre-trained networks

### 3. Data Augmentation

```python
from skimage import transform

# Augment training data
- Random rotations (±15°)
- Horizontal flips
- Brightness adjustments
- Zooming in/out
```

### 4. Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('hog_rbf', svm_rbf_hog),
    ('hog_linear', svm_linear_hog),
    ('pca_rbf', svm_rbf_pca)
], voting='soft')
```

### 5. Deep Learning Comparison

Compare SVM with:
- Simple CNN
- ResNet-50 (transfer learning)
- VGG16 (transfer learning)

---

## 📚 References

### Papers

1. **HOG Original Paper**: Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection.* CVPR.
2. **SVM Tutorial**: Cortes, C., & Vapnik, V. (1995). *Support-vector networks.* Machine Learning.
3. **PCA**: Pearson, K. (1901). *On lines and planes of closest fit to systems of points in space.*

### Documentation

- [Scikit-Learn SVM](https://scikit-learn.org/stable/modules/svm.html)
- [Scikit-Image HOG](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)
- [Scikit-Learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

### Tutorials

- [HOG Descriptor Tutorial](https://www.learnopencv.com/histogram-of-oriented-gradients/)
- [PCA Step-by-Step](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
- [SVM Explained](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)

### Datasets

- [Microsoft Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)
- [More Image Datasets](https://www.kaggle.com/datasets)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Performance Optimization**: Faster feature extraction
2. **More Feature Extractors**: SIFT, SURF, ORB
3. **Deep Learning Comparison**: Add CNN baseline
4. **Hyperparameter Tuning**: GridSearchCV implementation
5. **Documentation**: More examples and explanations

---

## 📝 License

This project is part of the Machine Learning repository. See the main repository LICENSE file.

---

## 📞 Contact

**Project Maintainer**: Pravin Kumar S

- 📧 Email: winnypine@gmail.com
- 💼 LinkedIn: [pravin-kumar-34b172216](https://www.linkedin.com/in/pravin-kumar-34b172216/)
- 🐙 GitHub: [prawin4E](https://github.com/prawin4E)

---

## 🎓 Educational Value

This project is ideal for:

- **Learning SVM**: Comprehensive implementation with explanations
- **Feature Engineering**: Comparing different extraction methods
- **Computer Vision**: Understanding HOG and image processing
- **Model Evaluation**: Using multiple metrics properly
- **Best Practices**: Error handling, documentation, visualization

---

## ⭐ Acknowledgments

- **Scikit-Learn**: Excellent ML library
- **Scikit-Image**: Computer vision tools
- **Microsoft Research**: Dogs vs Cats dataset
- **OpenCV**: Image processing library
- **Community**: Tutorials, papers, and Stack Overflow

---

<div align="center">

### 🐶🐱 Happy Classifying! 🚀

**If you find this project helpful, please give it a star! ⭐**

---

Made with ❤️ by Pravin Kumar S

*Last Updated: October 2025*

</div>
