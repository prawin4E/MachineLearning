# 🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)

A comprehensive machine learning project demonstrating **K-Nearest Neighbors (KNN)** algorithm for multi-class classification on the famous Iris dataset. This project includes detailed algorithm explanation, complete preprocessing pipeline, multiple distance metrics, and thorough performance analysis.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [What is K-Nearest Neighbors?](#-what-is-k-nearest-neighbors)
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Algorithm Variants](#-algorithm-variants)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Learning Resources](#-learning-resources)
- [Key Learnings](#-key-learnings)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project implements **K-Nearest Neighbors (KNN)** for multi-class classification on the Iris flower dataset. KNN is a simple yet powerful machine learning algorithm that classifies data points based on the classes of their nearest neighbors.

### What You'll Learn:
- **KNN Algorithm**: How it works and why it's effective
- **Distance Metrics**: Euclidean, Manhattan, Minkowski
- **Hyperparameter Tuning**: Finding optimal K value
- **Multi-class Classification**: Handling 3+ classes
- **Model Evaluation**: Classification metrics and visualization
- **Practical Applications**: Real-world use cases

---

## 🤔 What is K-Nearest Neighbors?

### Core Concept

**K-Nearest Neighbors (KNN)** is a **non-parametric, lazy learning** algorithm that classifies a data point based on how its neighbors are classified.

```
Simple Analogy:
"You are the average of the 5 people you spend the most time with"
→ KNN: "A data point belongs to the class most common among its K nearest neighbors"
```

### How KNN Works

```
Step 1: Choose K (number of neighbors to consider)
        ↓
Step 2: Calculate distance from new point to all training points
        ↓
Step 3: Find K nearest neighbors
        ↓
Step 4: Majority vote among these K neighbors
        ↓
Step 5: Assign the class with most votes
```

### Visual Example

```
Classify the "?" point with K=3:

    Class A (●)        Class B (■)

        ●                  ■
          ●    ?         ■
        ●  ●           ■

3 Nearest Neighbors: ● ● ●  (all Class A)
→ Prediction: Class A
```

---

## 📊 KNN vs Other Algorithms

| Aspect | KNN | Linear/Logistic Regression | Decision Trees |
|--------|-----|---------------------------|----------------|
| **Type** | Instance-based | Model-based | Tree-based |
| **Training** | No training (lazy) | Learns coefficients | Builds tree structure |
| **Prediction Speed** | Slow (compares all points) | Fast | Fast |
| **Interpretability** | Low | High | High |
| **Handles Non-linear** | Yes | No (linear only) | Yes |
| **Requires Scaling** | Yes (critical!) | Yes (for regularization) | No |
| **Memory Usage** | High (stores all data) | Low | Medium |
| **Best For** | Small-medium datasets | Large datasets, linear problems | Non-linear, interpretable |

---

## ✨ Features

### 🔧 Comprehensive Implementation
- ✅ Classic KNN algorithm
- ✅ Multiple distance metrics (Euclidean, Manhattan, Minkowski)
- ✅ Weighted and uniform voting
- ✅ K value optimization using cross-validation
- ✅ Feature scaling (StandardScaler)
- ✅ Train-test splitting with stratification

### 🛠️ Distance Metrics Explained

#### 1️⃣ **Euclidean Distance** (Default)
```
Formula: √[(x₁-x₂)² + (y₁-y₂)²]

Example: Distance between (1,2) and (4,6)
= √[(1-4)² + (2-6)²]
= √[9 + 16]
= √25 = 5

When to use:
✅ Most common choice
✅ When all features have similar scale
✅ General-purpose applications
```

#### 2️⃣ **Manhattan Distance** (City Block)
```
Formula: |x₁-x₂| + |y₁-y₂|

Example: Distance between (1,2) and (4,6)
= |1-4| + |2-6|
= 3 + 4 = 7

When to use:
✅ High-dimensional data
✅ Grid-like paths (like city blocks)
✅ When diagonal movement doesn't make sense
```

#### 3️⃣ **Minkowski Distance** (Generalized)
```
Formula: (Σ|xᵢ-yᵢ|ᵖ)^(1/p)

p=1: Manhattan distance
p=2: Euclidean distance
p=∞: Chebyshev distance

When to use:
✅ Want to experiment with different metrics
✅ Domain-specific distance requirements
```

### 📊 Visualization & EDA
- Feature distribution analysis
- Pairwise scatter plots
- Decision boundary visualization
- Confusion matrix heatmaps
- K value performance curves
- Error rate analysis

### 📈 Evaluation Metrics
- Accuracy Score
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- Classification Report
- Cross-validation scores
- K-value optimization curve

---

## 📁 Dataset

**Dataset**: Iris Flower Dataset

### Overview:
- **Source**: UCI Machine Learning Repository (built into scikit-learn)
- **Year**: 1936 (by Ronald Fisher)
- **Type**: Multi-class classification
- **Classes**: 3 species of Iris flowers

### Target Classes:
1. **Setosa** (Class 0) - Easy to distinguish
2. **Versicolor** (Class 1) - Medium difficulty
3. **Virginica** (Class 2) - Similar to Versicolor

### Features (4 measurements in cm):

| Feature | Description | Type | Example Range |
|---------|-------------|------|---------------|
| **sepal_length** | Length of sepal | Continuous | 4.3 - 7.9 cm |
| **sepal_width** | Width of sepal | Continuous | 2.0 - 4.4 cm |
| **petal_length** | Length of petal | Continuous | 1.0 - 6.9 cm |
| **petal_width** | Width of petal | Continuous | 0.1 - 2.5 cm |

### Botanical Terms:
```
Iris Flower Structure:

    Petal (colored)
         /\
        /  \
       |    |  ← Petal Length
        \  /
         \/
    ___________
    Sepal (green)
         |      ← Sepal Length
         |
```

### Dataset Statistics:
- **Total Samples**: 150 (50 per class)
- **Features**: 4 numerical features
- **Missing Values**: None
- **Class Distribution**: Perfectly balanced (50-50-50)
- **Linearly Separable**: Setosa is, but Versicolor & Virginica overlap

### Why Iris Dataset?

✅ **Perfect for Learning:**
- Small and manageable size
- Clean data (no missing values)
- Well-balanced classes
- Real-world measurements
- Non-trivial classification

✅ **Historical Importance:**
- One of the earliest datasets in ML
- Used by Ronald Fisher in 1936
- Standard benchmark for algorithms

✅ **Ideal for KNN:**
- Continuous features (distances make sense)
- Clear class separations (mostly)
- Demonstrates KNN strengths and limitations

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
Jupyter Notebook or JupyterLab
```

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd MachineLearning/Implementation/KNN
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Required Packages
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

Or use the main repository requirements:
```bash
cd ../../  # Go to repository root
pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Notebook

1. **Navigate to the KNN directory**
   ```bash
   cd MachineLearning/Implementation/KNN
   ```

2. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

3. **Open the notebook**
   - Click on `KNN.ipynb`
   - Run cells sequentially (Shift + Enter)

4. **Execute all cells**
   ```
   Cell → Run All
   ```

### Quick Start Code

```python
# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset (built into scikit-learn)
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features (IMPORTANT for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 🎓 Algorithm Variants

### 1️⃣ Basic KNN (Euclidean Distance, Uniform Weights)

**Standard implementation**

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train, y_train)
```

**Parameters:**
- `n_neighbors=5`: Use 5 nearest neighbors
- `weights='uniform'`: All neighbors have equal vote
- `metric='euclidean'`: Standard Euclidean distance

**When to Use:**
- ✅ First baseline model
- ✅ Features have similar scales (after scaling)
- ✅ All neighbors equally important

---

### 2️⃣ Weighted KNN (Distance-Based Weights)

**Closer neighbors have more influence**

```python
knn_weighted = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # Weight by inverse of distance
    metric='euclidean'
)
knn_weighted.fit(X_train, y_train)
```

**How It Works:**
```
Uniform Weights:        Distance Weights:
Each neighbor = 1 vote  Closer = More votes

Neighbor A (dist=1): 1  →  1.0 vote
Neighbor B (dist=2): 1  →  0.5 vote
Neighbor C (dist=3): 1  →  0.33 vote
```

**When to Use:**
- ✅ Closer neighbors more relevant
- ✅ Uneven neighbor distribution
- ✅ Better for regression problems

---

### 3️⃣ Manhattan Distance KNN

**City-block distance**

```python
knn_manhattan = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    metric='manhattan'  # L1 distance
)
knn_manhattan.fit(X_train, y_train)
```

**When to Use:**
- ✅ High-dimensional data
- ✅ Sparse features
- ✅ Grid-like problem structure

---

### 4️⃣ Optimized KNN (GridSearchCV)

**Find best hyperparameters automatically**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_:.4f}")
```

**Benefits:**
- ✅ Systematic hyperparameter search
- ✅ Cross-validation built-in
- ✅ Finds optimal configuration

---

## 🔧 Choosing the Right K Value

### The K Dilemma

| K Value | Behavior | Pros | Cons |
|---------|----------|------|------|
| **K=1** | Use only nearest neighbor | Simple, low bias | High variance, noisy |
| **K=3-5** | Common default | Balanced | Depends on data |
| **K=7-15** | Moderate smoothing | Reduces noise | May miss patterns |
| **K=Large** | Very smooth decision | Low variance | High bias, underfitting |

### Finding Optimal K

```python
# Test different K values
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# Plot results
plt.plot(k_range, k_scores)
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Optimal K Selection')
plt.show()

# Best K
optimal_k = k_range[np.argmax(k_scores)]
print(f"Optimal K: {optimal_k}")
```

### Rules of Thumb

1. **Start with K=√n** where n = number of training samples
2. **Use odd K** for binary classification (avoids ties)
3. **Use cross-validation** to find optimal K
4. **Consider computational cost** (larger K = slower)

---

## 📂 Project Structure

```
KNN/
│
├── KNN.ipynb                 # Main Jupyter notebook with implementation
├── README.md                 # This comprehensive documentation
│
└── DATASET_INSTRUCTIONS.md   # How to access Iris dataset
```

**Main Repository Structure:**
```
MachineLearning/
│
├── dataset/
│   ├── CAR DETAILS FROM CAR DEKHO.csv
│   ├── heart.csv
│   └── (Iris is built into scikit-learn)
│
├── Implementation/
│   ├── LinearRegression/
│   ├── LogisticRegression/
│   ├── SVM/
│   └── KNN/  ← You are here
│       ├── KNN.ipynb
│       ├── README.md
│       └── DATASET_INSTRUCTIONS.md
│
├── README.md
└── requirements.txt
```

---

## 📊 Results

### Expected Model Performance

*Run the notebook to see actual results*

| Model Variant | Accuracy | Notes |
|---------------|----------|-------|
| KNN (K=1) | ~95% | May overfit slightly |
| KNN (K=5, Euclidean) | ~97% | Good balance |
| KNN (K=5, Weighted) | ~98% | Slightly better |
| KNN (K=optimal) | ~98-100% | Best performance |

*Note: Iris is a well-separated dataset, so high accuracy is expected*

### Key Insights

1. **Feature Importance**: Petal length and width are most discriminative
2. **Class Separability**: Setosa is perfectly separated, Versicolor & Virginica have slight overlap
3. **Optimal K**: Usually between 3-7 for Iris dataset
4. **Scaling Impact**: Critical! Without scaling, accuracy drops significantly
5. **Distance Metrics**: Euclidean and Manhattan perform similarly for Iris

### Confusion Matrix Example

```
              Predicted
           Setosa  Versicolor  Virginica
Actual
Setosa        15      0          0       ✅ Perfect
Versicolor     0     14          1       ⚠️ 1 error
Virginica      0      0         15       ✅ Perfect

Overall Accuracy: 97.8%
```

---

## 📚 Learning Resources

### YouTube Tutorials

#### 🎥 KNN Fundamentals

1. **StatQuest: K-nearest neighbors, Clearly Explained**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=HVXime0nQeI)
   - 👤 By: StatQuest with Josh Starmer
   - ⏱️ Duration: 5 minutes
   - 📝 Topics: Core concept, visual explanation

2. **K-Nearest Neighbor Algorithm | KNN Algorithm | Data Science Tutorial**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=4HKqjENq9OU)
   - 👤 By: Simplilearn
   - ⏱️ Duration: 10 minutes
   - 📝 Topics: Algorithm walkthrough, use cases

3. **K Nearest Neighbors in Python - Machine Learning From Scratch 01**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=ngLyX54e1LU)
   - 👤 By: Python Engineer
   - ⏱️ Duration: 19 minutes
   - 📝 Topics: Implementation from scratch

#### 🎥 Python Implementation

4. **K-Nearest Neighbors (KNN) with Python | Scikit-Learn**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=6BTNu21ZHHI)
   - 👤 By: codebasics
   - ⏱️ Duration: 25 minutes
   - 📝 Topics: Practical implementation, Iris dataset

5. **Machine Learning Tutorial Python - 10: KNearestNeighbor Algorithm**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=4HKqjENq9OU)
   - 👤 By: codebasics
   - ⏱️ Duration: 15 minutes
   - 📝 Topics: Complete project walkthrough

#### 🎥 Advanced Topics

6. **Choosing K in K-Nearest Neighbors**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=4HKqjENq9OU)
   - 👤 By: Data Professor
   - ⏱️ Duration: 12 minutes
   - 📝 Topics: Hyperparameter tuning, cross-validation

7. **Distance Metrics in Machine Learning**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=QLVMqwpOLPk)
   - 👤 By: Krish Naik
   - ⏱️ Duration: 20 minutes
   - 📝 Topics: Euclidean, Manhattan, Cosine similarity

---

### Documentation & Articles

#### 📖 Official Documentation

1. **Scikit-Learn: KNeighborsClassifier**
   - 🔗 [Read Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
   - 📝 Complete API reference

2. **Scikit-Learn: Nearest Neighbors**
   - 🔗 [Read Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
   - 📝 Comprehensive guide to KNN variants

3. **Iris Dataset Documentation**
   - 🔗 [Read Documentation](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
   - 📝 Official dataset description

#### 📝 In-Depth Articles

4. **A Complete Guide to K-Nearest-Neighbors**
   - 🔗 [Read Article](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
   - 📰 Towards Data Science
   - 📝 Theory + implementation

5. **KNN Algorithm: When You Should Use It**
   - 🔗 [Read Article](https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/)
   - 📰 Machine Learning Mastery
   - 📝 Practical guide with examples

6. **Understanding Distance Metrics**
   - 🔗 [Read Article](https://towardsdatascience.com/9-distance-measures-in-data-science-918109d069fa)
   - 📰 Towards Data Science
   - 📝 Comprehensive distance metric comparison

7. **The Curse of Dimensionality in KNN**
   - 🔗 [Read Article](https://towardsdatascience.com/the-curse-of-dimensionality-50dc6e49aa1e)
   - 📰 Towards Data Science
   - 📝 KNN limitations in high dimensions

#### 🎓 Interactive Tutorials

8. **Kaggle: Introduction to Machine Learning**
   - 🔗 [Start Course](https://www.kaggle.com/learn/intro-to-machine-learning)
   - 📝 Hands-on KNN exercises

9. **DataCamp: KNN in Python**
   - 🔗 [Start Course](https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn)
   - 📝 Interactive Python tutorial

---

### 🎯 Recommended Learning Path

```
1. Start: Watch StatQuest KNN video (5 min)
           ↓
2. Theory: Read Scikit-Learn KNN documentation
           ↓
3. Practice: Run this notebook step-by-step
           ↓
4. Deep Dive: Learn about distance metrics
           ↓
5. Advanced: Study K value optimization
           ↓
6. Compare: Understand KNN vs other algorithms
           ↓
7. Project: Implement KNN on your own dataset
           ↓
8. Explore: Try KNN for regression problems
```

---

## 💡 Key Learnings

### 🎯 When to Use KNN?

| ✅ Good Use Cases | ❌ Not Recommended |
|------------------|-------------------|
| Small-medium datasets | Very large datasets |
| Multi-class classification | Real-time predictions needed |
| Recommendation systems | High-dimensional data (>20 features) |
| Pattern recognition | Imbalanced datasets |
| Anomaly detection | When interpretability is critical |
| Image classification (small) | Memory-constrained systems |

### 🔑 Best Practices

1. **Always Scale Your Features**
   ```python
   # BAD: Without scaling
   knn.fit(X_train, y_train)  # Features with different scales!

   # GOOD: With scaling
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   knn.fit(X_train_scaled, y_train)
   ```

2. **Choose Appropriate K Value**
   - Too small (K=1): Overfitting, sensitive to noise
   - Too large (K=n): Underfitting, loses local patterns
   - Use cross-validation to find optimal K

3. **Select Right Distance Metric**
   - Euclidean: Default, works for most cases
   - Manhattan: High-dimensional, sparse data
   - Minkowski: Experiment with different p values

4. **Handle Imbalanced Data**
   ```python
   # Use stratified sampling
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, stratify=y  # Maintains class distribution
   )
   ```

5. **Consider Computational Cost**
   - Training: O(1) - no training needed!
   - Prediction: O(n × d) - compare with all points
   - Use KD-trees or Ball-trees for large datasets:
   ```python
   knn = KNeighborsClassifier(algorithm='kd_tree')
   ```

### ⚠️ Common Pitfalls

1. **Forgetting to Scale**
   - Impact: Features with larger ranges dominate
   - Solution: Always use StandardScaler or MinMaxScaler

2. **Using Even K for Binary Classification**
   - Impact: Possible tie votes
   - Solution: Use odd K values (3, 5, 7, etc.)

3. **Not Handling Missing Values**
   - Impact: Distance calculations fail
   - Solution: Impute before applying KNN

4. **Ignoring Curse of Dimensionality**
   - Impact: All points become equidistant in high dimensions
   - Solution: Use dimensionality reduction (PCA) first

5. **Choosing K Without Validation**
   - Impact: Suboptimal performance
   - Solution: Use GridSearchCV or cross-validation

### 📊 KNN vs Other Classifiers

**Comparison Table:**

| Aspect | KNN | Logistic Regression | SVM | Decision Tree |
|--------|-----|---------------------|-----|---------------|
| **Training Time** | O(1) | O(n) | O(n²-n³) | O(n log n) |
| **Prediction Time** | O(n) | O(1) | O(1) | O(log n) |
| **Memory** | High | Low | Medium | Low |
| **Non-linear** | Yes | No | Yes (with kernel) | Yes |
| **Interpretability** | Low | High | Low | High |
| **Feature Scaling** | Required | Recommended | Required | Not needed |
| **Handles Noise** | Poor | Good | Good | Poor |

### 🎓 Advanced Tips

1. **Weighted Voting**
   ```python
   # Closer neighbors have more influence
   knn = KNeighborsClassifier(weights='distance')
   ```

2. **Custom Distance Functions**
   ```python
   def custom_distance(x, y):
       return np.sum(np.abs(x - y))

   knn = KNeighborsClassifier(metric=custom_distance)
   ```

3. **Dimensionality Reduction**
   ```python
   from sklearn.decomposition import PCA

   pca = PCA(n_components=2)
   X_reduced = pca.fit_transform(X)
   knn.fit(X_reduced, y)
   ```

4. **Feature Selection**
   ```python
   from sklearn.feature_selection import SelectKBest

   selector = SelectKBest(k=3)
   X_selected = selector.fit_transform(X, y)
   ```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contribution
- Add KNN for regression implementation
- Implement custom distance metrics
- Add visualization of decision boundaries
- Compare with other classifiers
- Add real-world dataset examples
- Create interactive Streamlit app
- Add performance optimization techniques

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Ronald Fisher (1936), UCI ML Repository
- **Libraries**: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
- **Inspiration**: Classical machine learning textbooks and courses
- **Community**: Data Science and ML community

---

## 📞 Contact

**Project Maintainer**: Pravin Kumar S

- 📧 Email: winnypine@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/pravin-kumar-34b172216/
- 🐙 GitHub: prawin4E

---

## 🔮 Future Enhancements

- [ ] Implement KNN for regression
- [ ] Add KD-tree and Ball-tree algorithms
- [ ] Implement weighted KNN variants
- [ ] Add custom distance metrics
- [ ] Create decision boundary visualizations
- [ ] Compare with SVM, Decision Trees, Neural Networks
- [ ] Add feature importance analysis
- [ ] Implement cross-validation strategies
- [ ] Create interactive Streamlit dashboard
- [ ] Add performance optimization techniques
- [ ] Deploy as REST API
- [ ] Add GPU acceleration for large datasets

---

## 📚 Additional Resources

### Books 📖

1. **"Introduction to Statistical Learning"** by Gareth James et al.
   - Chapter 2.2.3: K-Nearest Neighbors
   - Free PDF: http://faculty.marshall.usc.edu/gareth-james/ISL/

2. **"Pattern Recognition and Machine Learning"** by Christopher Bishop
   - Chapter 2.5: Nearest Neighbor Methods

3. **"The Elements of Statistical Learning"** by Hastie, Tibshirani, Friedman
   - Chapter 13: Prototype Methods and Nearest Neighbors
   - Free PDF: https://hastie.su.domains/ElemStatLearn/

### Research Papers 📄

1. **Original KNN Paper**: "Nearest neighbor pattern classification" by Cover & Hart (1967)
2. **Distance Metrics**: "A Survey of Distance and Similarity Measures" by Cha (2007)
3. **Curse of Dimensionality**: "When Is 'Nearest Neighbor' Meaningful?" by Beyer et al. (1999)

### Courses 🎓

1. **Andrew Ng's Machine Learning Course**
   - Coursera: https://www.coursera.org/learn/machine-learning

2. **Google's Machine Learning Crash Course**
   - Website: https://developers.google.com/machine-learning/crash-course

3. **Fast.ai Practical Deep Learning**
   - Website: https://www.fast.ai/

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star! ⭐

**Happy Learning! 🚀**

---

**Built with ❤️ for the ML Community**

Made by Pravin Kumar S

</div>
