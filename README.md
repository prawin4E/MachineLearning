# 🚗 Car Price Prediction using Linear Regression

A comprehensive machine learning project demonstrating various linear regression techniques for predicting used car prices. This project covers complete data preprocessing, feature engineering, multiple regression models, and detailed performance analysis.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Types](#-model-types)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Learning Resources](#-learning-resources)
  - [YouTube Tutorials](#youtube-tutorials)
  - [Documentation & Articles](#documentation--articles)
- [Key Learnings](#-key-learnings)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project implements a complete machine learning pipeline for predicting used car prices using various linear regression techniques. It demonstrates industry-standard practices in:

- **Data Preprocessing**: Handling missing values, duplicates, and outliers
- **Feature Engineering**: Creating meaningful features from raw data
- **Exploratory Data Analysis**: Comprehensive visualization and statistical analysis
- **Model Training**: Implementing 4 types of linear regression
- **Model Evaluation**: Comparing performance using multiple metrics
- **Best Practices**: Following ML workflow standards

---

## ✨ Features

### 🔧 Comprehensive Preprocessing
- ✅ Missing value imputation (median/mode strategies)
- ✅ Duplicate detection and removal
- ✅ Outlier detection using IQR method
- ✅ Outlier treatment via Winsorization
- ✅ Feature scaling using StandardScaler
- ✅ Train-test splitting (80-20)

### 🛠️ Feature Engineering
- 🏷️ Brand extraction from car names
- 📅 Car age calculation
- 📊 KM per year computation
- 🎯 Categorical binning for analysis
- 🔢 Label encoding for ordinal features
- 🎭 One-hot encoding for nominal features

### 📊 Visualization & EDA
- Distribution plots (histograms, box plots)
- Correlation heatmaps
- Q-Q plots for normality testing
- Scatter plots for relationships
- Residual analysis
- Feature importance visualization

### 🤖 Multiple Regression Models
1. **Linear Regression (OLS)** - Vanilla implementation
2. **Ridge Regression (L2)** - Handles multicollinearity
3. **Lasso Regression (L1)** - Automatic feature selection
4. **ElasticNet (L1+L2)** - Combined regularization

### 📈 Performance Metrics
- R² Score (Coefficient of Determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Cross-validation support

---

## 📁 Dataset

**Dataset**: CAR DETAILS FROM CAR DEKHO

### Features:
- **name**: Car model name
- **year**: Manufacturing year
- **selling_price**: Price in INR (Target Variable)
- **km_driven**: Total kilometers driven
- **fuel**: Fuel type (Petrol/Diesel/CNG/LPG)
- **seller_type**: Individual/Dealer/Trustmark Dealer
- **transmission**: Manual/Automatic
- **owner**: First/Second/Third/Fourth & Above

### Dataset Statistics:
- **Samples**: ~4,340 records
- **Features**: 8 original features
- **After Engineering**: 100+ features (post encoding)

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
cd LinearRegression
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

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Requirements.txt Content:
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
jupyter>=1.0.0
```

---

## 💻 Usage

### Running the Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**
   - Navigate to `Implementation/LinearRegression.ipynb`
   - Run cells sequentially from top to bottom

3. **Execute all cells**
   ```
   Cell → Run All
   ```

### Quick Start Code

```python
# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('dataset/CAR DETAILS FROM CAR DEKHO.csv')

# Preprocess (see notebook for complete pipeline)
# ... preprocessing steps ...

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
predictions = model.predict(X_test_scaled)
```

---

## 🎓 Model Types

### 1️⃣ Linear Regression (OLS)

**Ordinary Least Squares - The Vanilla Approach**

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

**When to Use:**
- ✅ Small to medium datasets
- ✅ Low multicollinearity
- ✅ Need interpretable results
- ✅ Baseline model comparison

**Pros:** Simple, fast, interpretable  
**Cons:** Prone to overfitting, sensitive to outliers

---

### 2️⃣ Ridge Regression (L2 Regularization)

**Adds L2 penalty: α × Σ(β²)**

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

**When to Use:**
- ✅ High multicollinearity
- ✅ Many correlated features
- ✅ Want to keep all features
- ✅ Prevent overfitting

**Pros:** Handles multicollinearity, reduces overfitting  
**Cons:** Keeps all features (no selection)

---

### 3️⃣ Lasso Regression (L1 Regularization)

**Adds L1 penalty: α × Σ|β|**

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=1.0)
model.fit(X_train, y_train)
```

**When to Use:**
- ✅ Many irrelevant features
- ✅ Need automatic feature selection
- ✅ Want sparse model
- ✅ High-dimensional data

**Pros:** Feature selection, interpretable, sparse models  
**Cons:** Unstable with correlated features

---

### 4️⃣ ElasticNet (L1 + L2)

**Combines both penalties: α × [l1_ratio × Σ|β| + (1-l1_ratio) × Σ(β²)]**

```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
model.fit(X_train, y_train)
```

**When to Use:**
- ✅ Multicollinearity + irrelevant features
- ✅ Best of both worlds
- ✅ Grouped feature selection
- ✅ Complex datasets

**Pros:** Stable, handles both issues, flexible  
**Cons:** Two hyperparameters to tune

---

## 📂 Project Structure

```
LinearRegression/
│
├── dataset/
│   └── CAR DETAILS FROM CAR DEKHO.csv    # Raw dataset
│
├── Implementation/
│   └── LinearRegression.ipynb            # Main Jupyter notebook
│
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
└── LICENSE                               # License file
```

---

## 📊 Results

### Model Performance Comparison

| Model | R² Score | RMSE | MAE | Features Used |
|-------|----------|------|-----|---------------|
| Linear Regression | 0.8XXX | ₹XX,XXX | ₹XX,XXX | All |
| Ridge Regression | 0.8XXX | ₹XX,XXX | ₹XX,XXX | All |
| Lasso Regression | 0.8XXX | ₹XX,XXX | ₹XX,XXX | Selected |
| ElasticNet | 0.8XXX | ₹XX,XXX | ₹XX,XXX | Selected |

*Note: Run the notebook to see actual performance metrics*

### Key Insights

1. **Feature Importance**: Year, brand, and transmission are key predictors
2. **Multicollinearity**: Moderate correlation between features
3. **Outliers**: Significant outliers in price and km_driven
4. **Model Selection**: All models perform similarly (well-structured data)

---

## 📚 Learning Resources

### YouTube Tutorials

#### 🎥 Linear Regression Fundamentals

1. **StatQuest: Linear Regression, Clearly Explained!!!**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=nk2CQITm_eo)
   - 👤 By: StatQuest with Josh Starmer
   - ⏱️ Duration: 27 minutes
   - 📝 Topics: OLS, R², intuitive explanations

2. **Khan Academy: Introduction to Linear Regression**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=yMgFHbjbAW8)
   - 👤 By: Khan Academy
   - ⏱️ Duration: 11 minutes
   - 📝 Topics: Basic concepts, least squares

#### 🎥 Regularization Techniques

3. **StatQuest: Regularization Part 1 - Ridge Regression**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=Q81RR3yKn30)
   - 👤 By: StatQuest with Josh Starmer
   - ⏱️ Duration: 20 minutes
   - 📝 Topics: L2 regularization, bias-variance tradeoff

4. **StatQuest: Regularization Part 2 - Lasso Regression**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=NGf0voTMlcs)
   - 👤 By: StatQuest with Josh Starmer
   - ⏱️ Duration: 8 minutes
   - 📝 Topics: L1 regularization, feature selection

5. **StatQuest: Ridge vs Lasso Regression, Visualized!!!**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=Xm2C_gTAl8c)
   - 👤 By: StatQuest with Josh Starmer
   - ⏱️ Duration: 9 minutes
   - 📝 Topics: Comparison, when to use which

6. **ElasticNet Regression Explained**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=1dKRdX9bfIo)
   - 👤 By: Normalized Nerd
   - ⏱️ Duration: 15 minutes
   - 📝 Topics: Combined regularization, tuning

#### 🎥 Python Implementation

7. **Python Machine Learning: Linear Regression with Scikit-Learn**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=1BYu65vLKdA)
   - 👤 By: Corey Schafer
   - ⏱️ Duration: 32 minutes
   - 📝 Topics: Scikit-learn, full implementation

8. **Complete Linear Regression Project in Python**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=TJveOYsK6MY)
   - 👤 By: Krish Naik
   - ⏱️ Duration: 1 hour
   - 📝 Topics: End-to-end project, EDA, deployment

#### 🎥 Feature Engineering & Preprocessing

9. **Feature Engineering for Machine Learning**
   - 📺 [Watch Video](https://www.youtube.com/watch?v=6WDFfaYtN6s)
   - 👤 By: Krish Naik
   - ⏱️ Duration: 45 minutes
   - 📝 Topics: Creating features, transformations

10. **Data Preprocessing in Machine Learning**
    - 📺 [Watch Video](https://www.youtube.com/watch?v=7PtFLfPyHqI)
    - 👤 By: Simplilearn
    - ⏱️ Duration: 30 minutes
    - 📝 Topics: Scaling, encoding, handling missing data

#### 🎥 Model Evaluation

11. **Understanding R-Squared, Adjusted R-Squared, and RMSE**
    - 📺 [Watch Video](https://www.youtube.com/watch?v=nRFCjZbSR4w)
    - 👤 By: Data School
    - ⏱️ Duration: 16 minutes
    - 📝 Topics: Evaluation metrics, interpretation

12. **Cross Validation Explained**
    - 📺 [Watch Video](https://www.youtube.com/watch?v=fSytzGwwBVw)
    - 👤 By: StatQuest with Josh Starmer
    - ⏱️ Duration: 6 minutes
    - 📝 Topics: K-fold CV, validation strategies

---

### Documentation & Articles

#### 📖 Official Documentation

1. **Scikit-Learn: Linear Models**
   - 🔗 [Read Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
   - 📝 Complete guide to all linear models in sklearn

2. **Scikit-Learn: Preprocessing**
   - 🔗 [Read Documentation](https://scikit-learn.org/stable/modules/preprocessing.html)
   - 📝 Data preprocessing techniques

3. **Pandas Documentation**
   - 🔗 [Read Documentation](https://pandas.pydata.org/docs/)
   - 📝 Data manipulation and analysis

#### 📝 In-Depth Articles

4. **Introduction to Linear Regression**
   - 🔗 [Read Article](https://towardsdatascience.com/introduction-to-linear-regression-in-python-c12a072bedf0)
   - 📰 Towards Data Science
   - 📝 Theory + Python implementation

5. **Ridge and Lasso Regression: A Complete Guide**
   - 🔗 [Read Article](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b)
   - 📰 Towards Data Science
   - 📝 Detailed comparison with examples

6. **ElasticNet Regression Explained**
   - 🔗 [Read Article](https://machinelearningmastery.com/elastic-net-regression-in-python/)
   - 📰 Machine Learning Mastery
   - 📝 Practical guide with code

7. **Feature Engineering Techniques**
   - 🔗 [Read Article](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)
   - 📰 Towards Data Science
   - 📝 Best practices and techniques

8. **Understanding the Bias-Variance Tradeoff**
   - 🔗 [Read Article](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
   - 📰 Towards Data Science
   - 📝 Why regularization works

#### 🎓 Interactive Tutorials

9. **Kaggle: Introduction to Machine Learning**
   - 🔗 [Start Course](https://www.kaggle.com/learn/intro-to-machine-learning)
   - 📝 Interactive Python notebooks

10. **Google's Machine Learning Crash Course**
    - 🔗 [Start Course](https://developers.google.com/machine-learning/crash-course)
    - 📝 Free course with TensorFlow examples

#### 📊 Cheat Sheets

11. **Scikit-Learn Cheat Sheet**
    - 🔗 [Download PDF](https://www.datacamp.com/cheat-sheet/scikit-learn-cheat-sheet-python-machine-learning)
    - 📝 Quick reference for sklearn functions

12. **Pandas Cheat Sheet**
    - 🔗 [Download PDF](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
    - 📝 Essential pandas operations

---

### 🎯 Recommended Learning Path

```
1. Start: Watch StatQuest Linear Regression video
           ↓
2. Theory: Read Scikit-Learn Linear Models documentation
           ↓
3. Practice: Follow Corey Schafer's implementation tutorial
           ↓
4. Deep Dive: Watch regularization videos (Ridge, Lasso)
           ↓
5. Advanced: Read articles on feature engineering
           ↓
6. Project: Work through this notebook!
           ↓
7. Deploy: Build your own car price predictor
```

---

## 💡 Key Learnings

### 🎯 When to Use Which Model?

| Scenario | Recommended Model |
|----------|------------------|
| **Few features, no correlation** | Linear Regression |
| **Many correlated features** | Ridge Regression |
| **Many features, some irrelevant** | Lasso Regression |
| **Both issues: correlation + irrelevant** | ElasticNet |
| **Unsure? Start with** | Linear Regression (baseline) |

### 🔑 Best Practices

1. **Always preprocess data thoroughly**
   - Handle missing values
   - Detect and treat outliers
   - Scale features
   - Encode categorical variables

2. **Perform EDA before modeling**
   - Understand data distribution
   - Check for correlations
   - Identify patterns and anomalies

3. **Compare multiple models**
   - Start with baseline (Linear Regression)
   - Try regularized versions
   - Use cross-validation
   - Select based on metrics + interpretability

4. **Monitor for overfitting**
   - Compare train vs test performance
   - Use regularization when needed
   - Simplify models when possible

5. **Document everything**
   - Keep track of experiments
   - Document preprocessing steps
   - Record model parameters
   - Save results for comparison

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
- Add more visualization techniques
- Implement cross-validation
- Add polynomial features
- Try ensemble methods
- Improve feature engineering
- Add model deployment code

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: CarDekho for providing the car price dataset
- **Libraries**: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
- **Inspiration**: Various Kaggle notebooks and tutorials
- **Community**: Stack Overflow and Data Science community

---

## 📞 Contact

**Project Maintainer**: Pravin Kumar S

- 📧 Email: [Your Email]
- 💼 LinkedIn: [Your LinkedIn]
- 🐙 GitHub: [Your GitHub]

---

## 🔮 Future Enhancements

- [ ] Add cross-validation implementation
- [ ] Implement polynomial regression
- [ ] Add ensemble methods (Random Forest, XGBoost)
- [ ] Create Flask/Streamlit web app
- [ ] Add model persistence (pickle/joblib)
- [ ] Implement GridSearchCV for hyperparameter tuning
- [ ] Add more feature engineering techniques
- [ ] Create interactive dashboards
- [ ] Add model interpretability (SHAP values)
- [ ] Deploy to cloud (AWS/Azure/GCP)

---

## 📚 Additional Resources

### Books 📖
1. **"An Introduction to Statistical Learning"** by Gareth James et al.
   - Free PDF: http://faculty.marshall.usc.edu/gareth-james/ISL/

2. **"Hands-On Machine Learning"** by Aurélien Géron
   - GitHub: https://github.com/ageron/handson-ml2

### Courses 🎓
1. **Andrew Ng's Machine Learning Course**
   - Coursera: https://www.coursera.org/learn/machine-learning

2. **Fast.ai Practical Deep Learning**
   - Website: https://www.fast.ai/

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star! ⭐

**Happy Learning! 🚀**

---

Made with ❤️ by Pravin Kumar S

</div>
