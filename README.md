# 🤖 Machine Learning Projects

A comprehensive collection of machine learning implementations covering various algorithms and techniques. This repository serves as a practical learning resource with complete implementations, detailed documentation, and real-world datasets.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [About](#-about)
- [Implementations](#-implementations)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Datasets](#-datasets)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About

This repository contains hands-on implementations of various machine learning algorithms, demonstrating:

- ✅ Complete data preprocessing pipelines
- ✅ Feature engineering techniques
- ✅ Model training and evaluation
- ✅ Visualization and analysis
- ✅ Best practices in ML workflow
- ✅ Real-world datasets and use cases

Each implementation is organized in its own folder with dedicated documentation, making it easy to understand and learn from individual projects.

---

## 🎓 What's Covered

This repository spans the full spectrum of machine learning:

### 📊 Classical Machine Learning
- **Supervised Learning**: Linear/Logistic Regression, KNN, SVM
- **Unsupervised Learning**: K-Means Clustering
- **Feature Engineering**: HOG, PCA, StandardScaler

### 🧠 Deep Learning
- **Fundamentals**: Artificial Neural Networks (ANN)
- **Computer Vision**: Convolutional Neural Networks (CNN)
- **Sequence Processing**: RNN, BiRNN, LSTM
- **Applications**: Image Classification, Sentiment Analysis

### 🎯 Real-World Applications
- Car Price Prediction
- Heart Disease Classification
- Customer Segmentation
- Image Classification (Dogs vs Cats)
- Sentiment Analysis (Movie Reviews)
- Handwritten Digit Recognition

---

## 📚 Implementations

### 1️⃣ Linear Regression
**📁 [Implementation/LinearRegression](Implementation/LinearRegression/)**

A comprehensive car price prediction project demonstrating various **regression** techniques for predicting continuous values.

**Topics Covered:**
- Linear Regression (OLS)
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)
- ElasticNet (L1 + L2)
- Feature Engineering
- Data Preprocessing
- Model Evaluation

**Dataset:** Car Details from CarDekho
**Type:** Regression (Continuous Prediction)
**Status:** ✅ Complete

[📖 View Full Documentation →](Implementation/LinearRegression/README.md)

---

### 2️⃣ Logistic Regression
**📁 [Implementation/LogisticRegression](Implementation/LogisticRegression/)**

A comprehensive heart disease prediction project demonstrating **classification** techniques for binary prediction.

**Topics Covered:**
- Logistic Regression (Basic)
- L2 Regularization (Ridge)
- L1 Regularization (Lasso)
- Classification Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion Matrix Analysis
- Cross-Validation
- Hyperparameter Tuning
- **Detailed comparison with Linear Regression**

**Dataset:** Heart Disease UCI Dataset
**Type:** Classification (Binary Prediction)
**Status:** ✅ Complete

[📖 View Full Documentation →](Implementation/LogisticRegression/README.md)

---

### 3️⃣ K-Nearest Neighbors (KNN)
**📁 [Implementation/KNN](Implementation/KNN/)**

A comprehensive iris flower classification project demonstrating **K-Nearest Neighbors** algorithm for multi-class classification.

**Topics Covered:**
- KNN Algorithm (instance-based learning)
- Distance Metrics (Euclidean, Manhattan, Minkowski)
- K Value Optimization
- Multi-class Classification
- Weighted vs Uniform Voting
- Cross-Validation

**Dataset:** Iris Flower Dataset (UCI ML Repository)
**Type:** Classification (Multi-class)
**Status:** ✅ Complete

[📖 View Full Documentation →](Implementation/KNN/README.md)

---

### 4️⃣ Support Vector Machines (SVM)
**📁 [Implementation/SVM](Implementation/SVM/)**

Support Vector Machine implementations for various classification tasks.

**Topics Covered:**
- Linear SVM
- RBF Kernel SVM
- Polynomial Kernel SVM
- GPU-accelerated SVM
- Hyperparameter tuning

**Dataset:** Various classification datasets
**Type:** Classification
**Status:** ✅ Complete

[📖 View Full Documentation →](Implementation/SVM/README.md)

---

### 5️⃣ K-Means Clustering
**📁 [Implementation/KMeansClustering](Implementation/KMeansClustering/)**

A comprehensive unsupervised learning project demonstrating **K-Means Clustering** with **Elbow Method** for customer segmentation.

**Topics Covered:**
- K-Means Clustering Algorithm
- Elbow Method for Optimal K Selection
- Silhouette Score Analysis
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Customer Segmentation
- Cluster Profiling and Business Insights
- 2D and 3D Cluster Visualization

**Dataset:** Mall Customers Dataset
**Type:** Unsupervised Learning (Clustering)
**Status:** ✅ Complete

[📖 View Full Documentation →](Implementation/KMeansClustering/README.md)

---

### 6️⃣ SVM Image Classification (Dog vs Cat)
**📁 [Implementation/SVM_ImageClassification](Implementation/SVM_ImageClassification/)**

A **comprehensive comparison** of feature extraction techniques for image classification using SVM.

**Topics Covered:**
- HOG (Histogram of Oriented Gradients) Feature Extraction
- PCA (Principal Component Analysis) Dimensionality Reduction
- Linear, RBF, and Polynomial SVM Kernels
- 8+ Evaluation Metrics
- Image Preprocessing Pipeline
- Detailed Feature Engineering Analysis

**Dataset:** Microsoft Dogs vs Cats (25,000 images)
**Type:** Binary Image Classification
**Status:** ✅ Complete

[📖 View Full Documentation →](Implementation/SVM_ImageClassification/README.md)

---

## 🧠 Deep Learning Implementations

### 7️⃣ Artificial Neural Networks (ANN)
**📁 [Implementation/ANN](Implementation/ANN/)**

Foundation of deep learning - fully connected neural networks for tabular data.

**Topics Covered:**
- Sequential Model Architecture
- Dense (Fully Connected) Layers
- ReLU and Sigmoid Activation Functions
- Dropout Regularization
- Adam Optimizer
- Early Stopping
- Binary Classification

**Architecture:** 3 Hidden Layers (64→32→16 neurons)
**Type:** Binary Classification
**Status:** ✅ Complete

[📖 View Full Documentation →](Implementation/ANN/README.md)

---

### 8️⃣ Convolutional Neural Networks (CNN)
**📁 [Implementation/CNN](Implementation/CNN/)**

State-of-the-art architecture for **image classification** and computer vision tasks.

**Topics Covered:**
- Convolutional Layers (Feature Extraction)
- Pooling Layers (Dimensionality Reduction)
- Batch Normalization
- Data Augmentation
- Transfer Learning Concepts
- Filter Visualization

**Dataset:** MNIST (70,000 handwritten digits)
**Architecture:** 4 Conv Layers + 2 Dense Layers
**Type:** Multi-class Image Classification
**Status:** ✅ Complete

[📖 View Full Documentation →](Implementation/CNN/README.md)

---

### 9️⃣ Recurrent Neural Networks (RNN)
**📁 [Implementation/RNN](Implementation/RNN/)**

Sequential data processing for **sentiment analysis** and text classification.

**Topics Covered:**
- SimpleRNN Layers
- Embedding Layers
- Text Tokenization and Padding
- Sentiment Analysis
- Backpropagation Through Time (BPTT)
- Sequence Processing

**Dataset:** IMDB Movie Reviews (50,000 reviews)
**Architecture:** Embedding → 2 RNN Layers → Dense
**Type:** Binary Sentiment Classification
**Status:** ✅ Complete

[📖 View Full Documentation →](Implementation/RNN/README.md)

---

### 🔟 Bidirectional RNN (BiRNN)
**📁 [Implementation/BiRNN](Implementation/BiRNN/)**

Enhanced RNN that processes sequences in **both forward and backward** directions for better context understanding.

**Topics Covered:**
- Bidirectional Processing
- Forward and Backward RNN
- Context from Both Directions
- Improved Accuracy

**Type:** Sequence Classification
**Status:** ✅ Complete

[📖 View Full Documentation →](Implementation/BiRNN/README.md)

---

### 1️⃣1️⃣ Long Short-Term Memory (LSTM)
**📁 [Implementation/LSTM](Implementation/LSTM/)**

Advanced RNN variant that solves the vanishing gradient problem for long sequences.

**Topics Covered:**
- LSTM Gates (Input, Forget, Output)
- Long-term Dependencies
- Cell State Management

**Type:** Sequence Processing
**Status:** 🚧 In Progress

---

## 🔮 Coming Soon

More implementations will be added:

- [ ] BiLSTM (Bidirectional LSTM)
- [ ] GRU (Gated Recurrent Units)
- [ ] Decision Trees
- [ ] Random Forest
- [ ] Gradient Boosting (XGBoost, LightGBM)
- [ ] Naive Bayes
- [ ] Hierarchical Clustering
- [ ] DBSCAN Clustering
- [ ] Time Series Analysis
- [ ] Transformer Models
- [ ] GANs (Generative Adversarial Networks)

---

## 📂 Project Structure

```
MachineLearning/
│
├── dataset/                                    # Shared datasets
│   ├── CAR DETAILS FROM CAR DEKHO.csv          # Car price dataset
│   ├── heart.csv                               # Heart disease dataset
│   ├── Mall_Customers.csv                      # Mall customers dataset (optional)
│   └── PetImages/                              # Dogs vs Cats images (25k+ images)
│
├── Implementation/                             # All implementations
│   │
│   ├── LinearRegression/                       # 📊 Regression
│   │   ├── LinearRegression.ipynb
│   │   ├── README.md
│   │   └── Notes.md
│   │
│   ├── LogisticRegression/                     # 📊 Classification
│   │   ├── LogisticRegression.ipynb
│   │   ├── README.md
│   │   └── DATASET_INSTRUCTIONS.md
│   │
│   ├── KNN/                                    # 📊 Classification
│   │   ├── KNN.ipynb
│   │   ├── README.md
│   │   └── DATASET_INSTRUCTIONS.md
│   │
│   ├── SVM/                                    # 📊 Classification
│   │   ├── SVM.ipynb
│   │   ├── README.md
│   │   ├── DATASET_INSTRUCTIONS.md
│   │   ├── GPU_SVM_Example.py
│   │   └── GPU_Image_SVM_Example.py
│   │
│   ├── KMeansClustering/                       # 🎯 Clustering
│   │   ├── KMeansClustering.ipynb
│   │   ├── kmeans_script.py
│   │   ├── README.md
│   │   └── DATASET_INSTRUCTIONS.md
│   │
│   ├── SVM_ImageClassification/                # 🖼️ Image Classification
│   │   ├── SVM_DogVsCat.ipynb
│   │   └── README.md (1200+ lines)
│   │
│   ├── ANN/                                    # 🧠 Deep Learning
│   │   ├── ANN.ipynb
│   │   └── README.md
│   │
│   ├── CNN/                                    # 🧠 Deep Learning
│   │   ├── CNN.ipynb
│   │   └── README.md
│   │
│   ├── RNN/                                    # 🧠 Deep Learning
│   │   ├── RNN.ipynb
│   │   └── README.md
│   │
│   ├── BiRNN/                                  # 🧠 Deep Learning
│   │   ├── BiRNN.ipynb
│   │   └── README.md
│   │
│   └── LSTM/                                   # 🧠 Deep Learning
│       └── LSTM.ipynb
│
├── README.md                                   # This file
└── requirements.txt                            # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8 or higher
pip (Python package manager)
Jupyter Notebook or JupyterLab
```

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd MachineLearning
   ```

2. **Create virtual environment (Recommended)**
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Navigate to any implementation**
   - Go to `Implementation/[Algorithm]/`
   - Open the `.ipynb` file
   - Run cells sequentially

---

## 📁 Datasets

All datasets used in this repository are stored in the `dataset/` folder:

| Dataset | Used In | Size | Description |
|---------|---------|------|-------------|
| CAR DETAILS FROM CAR DEKHO.csv | Linear Regression | ~4,340 records | Used car pricing data with features like year, km_driven, fuel type, etc. |
| heart.csv | Logistic Regression | ~300 records | Heart disease UCI dataset with 13 clinical attributes for binary classification |

---

## 💻 Quick Start

### Running an Implementation

**Linear Regression (Continuous Prediction):**
```bash
cd Implementation/LinearRegression
jupyter notebook LinearRegression.ipynb
```

**Logistic Regression (Binary Classification):**
```bash
cd Implementation/LogisticRegression
jupyter notebook LogisticRegression.ipynb
```

### Using Virtual Environment

Each implementation can be run independently using the shared requirements.txt:

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific packages
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

---

## 📚 Learning Resources

Each implementation folder contains:
- 📓 **Jupyter Notebook**: Complete code with explanations
- 📖 **README.md**: Detailed documentation with theory
- 🎥 **Video Links**: Recommended YouTube tutorials
- 📝 **Articles**: Helpful blog posts and documentation
- 💡 **Best Practices**: When and how to use each technique

---

## 🛠️ Technologies Used

### Classical Machine Learning:
- **Python**: Primary programming language
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization and plotting
- **Scikit-Learn**: Classical ML algorithms and tools
- **SciPy**: Scientific computing
- **Scikit-Image**: Image processing (HOG, filters)
- **OpenCV**: Computer vision operations

### Deep Learning:
- **TensorFlow**: Deep learning framework
- **Keras**: High-level neural networks API
- **GPU Support**: CUDA acceleration for training

### Development Tools:
- **Jupyter**: Interactive notebooks for experimentation
- **Git**: Version control

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Add new implementations**
   - Create a new folder in `Implementation/`
   - Add notebook with complete code
   - Include comprehensive README
   - Follow existing structure

2. **Improve existing implementations**
   - Enhance visualizations
   - Add more techniques
   - Improve documentation
   - Fix bugs or issues

3. **Submit Pull Request**
   ```bash
   git checkout -b feature/NewImplementation
   git commit -m 'Add new implementation'
   git push origin feature/NewImplementation
   ```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Datasets**: Various open-source datasets from Kaggle and other platforms
- **Libraries**: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
- **Community**: Stack Overflow, Towards Data Science, and the ML community
- **Inspiration**: Various tutorials, courses, and research papers

---

## 📞 Contact

**Project Maintainer**: Pravin Kumar S

- 📧 Email: winnypine@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/pravin-kumar-34b172216/
- 🐙 GitHub: prawin4E

---

## 🎯 Learning Path

### For Beginners - Classical ML:
```
1. Start: Linear Regression (supervised learning basics)
         ↓
2. Classification: Logistic Regression
         ↓
3. Advanced: KNN, SVM (distance & kernel methods)
         ↓
4. Unsupervised: K-Means Clustering
         ↓
5. Practice: Apply to your own datasets
```

### For Deep Learning:
```
1. Foundation: ANN (neural network basics)
         ↓
2. Computer Vision: CNN (image classification)
         ↓
3. Sequences: RNN (text & time series)
         ↓
4. Advanced: BiRNN, LSTM (long-term dependencies)
         ↓
5. Projects: Build your own deep learning models
```

### Complete Journey:
```
Classical ML → Feature Engineering → Deep Learning → Production
```

---

## 📈 Repository Stats

- **Total Implementations**: 11 (and growing!)
- **Total Notebooks**: 11+
- **Lines of Code**: 15,000+
- **Documentation Pages**: 6,000+
- **Algorithms Covered**: 
  - **Classical ML**: Regression, Classification, Clustering
  - **Deep Learning**: ANN, CNN, RNN, BiRNN, LSTM
  - **Computer Vision**: Image Classification with SVM, CNN
  - **NLP**: Sentiment Analysis with RNN

---

<div align="center">

### ⭐ If you find this repository helpful, please give it a star! ⭐

**Happy Learning! 🚀**

---

Made with ❤️ by Pravin Kumar S

</div>

