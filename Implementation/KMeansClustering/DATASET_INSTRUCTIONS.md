# 📁 Dataset Instructions - Mall Customers Dataset

## Dataset Overview

This project uses the **Mall Customers Dataset** for demonstrating K-Means Clustering and the Elbow Method.

---

## 🎯 Dataset Information

| Property | Details |
|----------|---------|
| **Name** | Mall Customers Dataset |
| **Source** | Kaggle / Sample Dataset |
| **Size** | 200 samples |
| **Features** | 5 columns |
| **Target** | No labels (unsupervised learning) |
| **Type** | Customer segmentation data |
| **Format** | CSV file |

---

## 📊 Features Description

### Columns:

1. **CustomerID** (Integer)
   - Unique identifier for each customer
   - Range: 1 to 200
   - Not used in clustering

2. **Gender** (Categorical)
   - Customer gender
   - Values: Male, Female
   - Can be used for analysis but not primary clustering feature

3. **Age** (Integer)
   - Customer age in years
   - Range: 18 to 70 years
   - Useful for demographic analysis

4. **Annual Income (k$)** (Integer)
   - Annual income in thousands of dollars
   - Range: 15k$ to 137k$
   - **Primary feature for clustering**

5. **Spending Score (1-100)** (Integer)
   - Score assigned by mall based on customer behavior
   - Range: 1 to 100
   - **Primary feature for clustering**

---

## 💾 How to Get the Dataset

### Option 1: Download from Kaggle

1. **Visit Kaggle:**
   ```
   https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
   ```

2. **Download:**
   - Click "Download" button
   - Save as `Mall_Customers.csv`

3. **Place in project:**
   ```bash
   # Save to dataset folder
   MachineLearning/dataset/Mall_Customers.csv
   ```

---

### Option 2: Use Synthetic Data (Generated in Notebook)

The notebook includes code to generate similar synthetic data if you don't have the original dataset:

```python
import numpy as np
import pandas as pd

# Generate synthetic mall customers data
np.random.seed(42)

n_samples = 200

data = {
    'CustomerID': range(1, n_samples + 1),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Age': np.random.randint(18, 71, n_samples),
    'Annual Income (k$)': np.random.randint(15, 138, n_samples),
    'Spending Score (1-100)': np.random.randint(1, 100, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('Mall_Customers.csv', index=False)
```

---

### Option 3: Copy Sample Data

Create a CSV file with this sample data structure:

```csv
CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
...
```

---

## 📂 File Placement

### Recommended Structure:

```
MachineLearning/
│
├── dataset/                        # Shared datasets folder
│   ├── CAR DETAILS FROM CAR DEKHO.csv
│   ├── heart.csv
│   └── Mall_Customers.csv          ← Place dataset here
│
├── Implementation/
│   └── KMeansClustering/
│       ├── KMeansClustering.ipynb
│       ├── README.md
│       └── DATASET_INSTRUCTIONS.md  ← You are here
```

### Loading in Notebook:

```python
# Load from shared dataset folder
df = pd.read_csv('../../dataset/Mall_Customers.csv')

# Or load from current directory (if placed locally)
df = pd.read_csv('Mall_Customers.csv')
```

---

## 🔍 Dataset Verification

### Check if dataset is loaded correctly:

```python
import pandas as pd

# Load dataset
df = pd.read_csv('Mall_Customers.csv')

# Verify shape
print(f"Dataset shape: {df.shape}")
# Expected: (200, 5)

# Check columns
print(f"Columns: {df.columns.tolist()}")
# Expected: ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Check for missing values
print(f"Missing values: {df.isnull().sum().sum()}")
# Expected: 0

# Preview data
print(df.head())
```

### Expected Output:

```
Dataset shape: (200, 5)
Columns: ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
Missing values: 0

   CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40
```

---

## ✅ Dataset Quality Checks

### 1. Check Data Types:
```python
print(df.dtypes)
```

### 2. Check Value Ranges:
```python
print(df.describe())
```

### 3. Check for Duplicates:
```python
print(f"Duplicates: {df.duplicated().sum()}")
```

### 4. Check Class Distribution:
```python
print(df['Gender'].value_counts())
```

---

## 🎯 Features Used for Clustering

### Primary Features (2D Clustering):

1. **Annual Income (k$)**
   - Represents customer's financial capacity
   - Continuous numerical feature
   - Needs scaling

2. **Spending Score (1-100)**
   - Represents spending behavior
   - Continuous numerical feature
   - Already normalized (1-100 scale)

### Why These Two Features?

✅ **Continuous numerical data** - Perfect for K-Means
✅ **Business relevance** - Income + Spending = Customer value
✅ **Clear patterns** - Natural customer segments emerge
✅ **Easy visualization** - 2D scatter plots
✅ **Actionable insights** - Direct marketing implications

---

## 📈 Expected Customer Segments

When clustering with Income vs Spending Score, expect these segments:

| Cluster | Income | Spending | Description | Marketing Strategy |
|---------|--------|----------|-------------|-------------------|
| **1** | Low | Low | Careful shoppers | Discounts, value deals |
| **2** | Low | High | Aspirational buyers | Credit options, installments |
| **3** | Medium | Medium | Average customers | Standard promotions |
| **4** | High | Low | Potential targets | Premium products, convince |
| **5** | High | High | VIP customers | Loyalty programs, exclusive |

---

## 🔧 Data Preprocessing Steps

### 1. Load Data:
```python
df = pd.read_csv('Mall_Customers.csv')
```

### 2. Select Features:
```python
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
```

### 3. Scale Features:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 4. Apply K-Means:
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
```

---

## ❓ Troubleshooting

### Issue 1: File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'Mall_Customers.csv'
```

**Solution:**
- Check file path
- Use absolute path or correct relative path
- Verify file name (case-sensitive on Linux/Mac)

### Issue 2: Column Names Mismatch
```
KeyError: 'Annual Income (k$)'
```

**Solution:**
```python
# Check actual column names
print(df.columns.tolist())

# Use correct names or rename
df.columns = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
```

### Issue 3: Data Type Issues
```
TypeError: ufunc 'isfinite' not supported for input types
```

**Solution:**
```python
# Convert to numeric
df['Annual Income (k$)'] = pd.to_numeric(df['Annual Income (k$)'], errors='coerce')
df['Spending Score (1-100)'] = pd.to_numeric(df['Spending Score (1-100)'], errors='coerce')
```

---

## 🔗 Additional Resources

### Download Links:

1. **Kaggle Dataset:**
   - https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

2. **UCI Machine Learning Repository:**
   - https://archive.ics.uci.edu/ml/index.php

3. **Alternative Datasets for K-Means:**
   - Iris Dataset (built into scikit-learn)
   - Wine Dataset (built into scikit-learn)
   - Wholesale Customers Dataset (UCI)

---

## 📝 Citation

If using this dataset in publications, please cite:

```
Mall Customers Dataset
Source: Kaggle
Author: Vijay Choudhary
URL: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
Year: 2018
```

---

## 💡 Tips

1. **Always verify data integrity** before clustering
2. **Scale features** for better clustering results
3. **Visualize data** before applying K-Means
4. **Check for outliers** that might affect clustering
5. **Validate clusters** using multiple metrics

---

## 🆘 Need Help?

If you encounter issues with the dataset:

1. **Check the notebook** - It includes fallback synthetic data generation
2. **Review error messages** - They usually point to the issue
3. **Verify file paths** - Use absolute paths if needed
4. **Contact maintainer** - See README.md for contact info

---

<div align="center">

**Ready to start clustering? Open the notebook! 🚀**

[← Back to README](README.md) | [Open Notebook →](KMeansClustering.ipynb)

</div>

