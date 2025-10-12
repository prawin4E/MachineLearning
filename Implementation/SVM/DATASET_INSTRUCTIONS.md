# 📁 Dataset Instructions for SVM Project

## 🎯 Required Dataset

This SVM project uses the **Heart Disease UCI Dataset** for binary classification.

---

## 📥 Download Instructions

### Option 1: Kaggle (Recommended)

1. **Visit Kaggle Dataset Page**
   - 🔗 **Link**: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

2. **Download the Dataset**
   - Click the **"Download"** button
   - You'll get a zip file containing `heart.csv`

3. **Extract and Place**
   - Extract the `heart.csv` file
   - Place it in the correct location (see below)

### Option 2: UCI Repository (Alternative)

1. **Visit UCI ML Repository**
   - 🔗 **Link**: [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

2. **Download Data**
   - Download the processed Cleveland data
   - Convert to CSV format if needed

---

## 📂 File Placement

### Correct Directory Structure

```
MachineLearning/
├── dataset/
│   ├── CAR DETAILS FROM CAR DEKHO.csv    # For Linear Regression
│   └── heart.csv                          # ← Place the heart dataset here
├── Implementation/
│   ├── LinearRegression/
│   ├── LogisticRegression/
│   └── SVM/
│       ├── SVM.ipynb                      # ← This notebook will read from ../../dataset/heart.csv
│       ├── README.md
│       └── DATASET_INSTRUCTIONS.md       # ← You are here
└── requirements.txt
```

### ⚠️ Important Path Notes

- The SVM notebook expects the dataset at: `../../dataset/heart.csv`
- This is a **relative path** from the SVM folder
- Make sure the file is named exactly `heart.csv`

---

## 📊 Dataset Information

### Basic Details
- **File Name**: `heart.csv`
- **Format**: CSV (Comma-Separated Values)
- **Size**: ~15 KB
- **Rows**: ~1,025 (before duplicate removal)
- **Columns**: 14 (13 features + 1 target)

### Target Variable
- **Column Name**: `target`
- **Type**: Binary classification
- **Values**:
  - `0` = No heart disease
  - `1` = Heart disease present

### Features (13 attributes)

| Column | Feature Name | Description | Type | Range |
|--------|--------------|-------------|------|-------|
| 1 | `age` | Age in years | Numerical | 29-77 |
| 2 | `sex` | Sex (1=male, 0=female) | Binary | 0, 1 |
| 3 | `cp` | Chest pain type | Categorical | 0-3 |
| 4 | `trestbps` | Resting blood pressure | Numerical | 94-200 |
| 5 | `chol` | Serum cholesterol in mg/dl | Numerical | 126-564 |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | Binary | 0, 1 |
| 7 | `restecg` | Resting electrocardiographic results | Categorical | 0-2 |
| 8 | `thalach` | Maximum heart rate achieved | Numerical | 71-202 |
| 9 | `exang` | Exercise induced angina | Binary | 0, 1 |
| 10 | `oldpeak` | ST depression induced by exercise | Numerical | 0-6.2 |
| 11 | `slope` | Slope of the peak exercise ST segment | Categorical | 0-2 |
| 12 | `ca` | Number of major vessels (0-3) | Numerical | 0-3 |
| 13 | `thal` | Thalium stress test result | Categorical | 1-3 |
| 14 | `target` | **Target variable** | Binary | 0, 1 |

---

## 🔍 Data Quality Checks

### Expected Data Issues
The notebook handles these automatically:

1. **Duplicates**: ~723 duplicate rows (will be removed)
2. **Missing Values**: None expected
3. **Outliers**: Present but medically significant (kept)
4. **Data Types**: All numerical, no string processing needed

### Sample Data Preview
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
41,0,1,130,204,0,0,172,0,1.4,2,0,2,1
```

---

## ✅ Verification Steps

### 1. Check File Location
```bash
# Navigate to your project root
cd MachineLearning

# Verify the file exists
ls -la dataset/heart.csv
```

### 2. Quick Data Check
```python
import pandas as pd

# Load and check the dataset
df = pd.read_csv('dataset/heart.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Target values: {df['target'].unique()}")
```

### Expected Output:
```
Shape: (1025, 14)
Columns: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
Target values: [1 0]
```

---

## 🚨 Troubleshooting

### Problem: "File not found" error

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: '../../dataset/heart.csv'
```

**Solutions:**
1. **Check file location**: Ensure `heart.csv` is in the `dataset/` folder
2. **Check file name**: Must be exactly `heart.csv` (case-sensitive)
3. **Check current directory**: Run notebook from `Implementation/SVM/` folder

### Problem: "Permission denied" error

**Solutions:**
1. **Check file permissions**: Make sure the file is readable
2. **Try absolute path**: Use full path instead of relative path

### Problem: Wrong dataset format

**Solutions:**
1. **Verify columns**: Should have exactly 14 columns
2. **Check separator**: Should be comma-separated (CSV)
3. **Re-download**: Get fresh copy from Kaggle

---

## 🔄 Alternative Datasets

If you want to experiment with other datasets:

### 1. Breast Cancer Dataset (Built-in)
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
# No download required - built into scikit-learn
```

### 2. Diabetes Dataset
- **Source**: [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
- **Place in**: `dataset/diabetes.csv`
- **Modify notebook**: Change file path in loading cell

### 3. Titanic Dataset
- **Source**: [Kaggle Titanic](https://www.kaggle.com/c/titanic/data)
- **Place in**: `dataset/titanic.csv`
- **Note**: Requires more preprocessing (categorical variables)

---

## 📝 Dataset Citation

If you use this dataset in research or publications:

```
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
School of Information and Computer Science.

Heart Disease Dataset. (1988). UCI Machine Learning Repository.
```

---

## 🆘 Need Help?

### Quick Checklist
- [ ] Downloaded `heart.csv` from Kaggle
- [ ] Placed file in `MachineLearning/dataset/` folder
- [ ] File is named exactly `heart.csv`
- [ ] Running notebook from `Implementation/SVM/` directory
- [ ] File has 1025 rows and 14 columns

### Still Having Issues?

1. **Check the main README**: `MachineLearning/README.md`
2. **Verify Python environment**: All required packages installed
3. **Try absolute path**: Replace `../../dataset/heart.csv` with full path
4. **Re-download dataset**: Get fresh copy from Kaggle

---

## 📞 Contact

If you continue to have dataset issues:

**Project Maintainer**: Pravin Kumar S
- 📧 Email: winnypine@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/pravin-kumar-34b172216/

---

<div align="center">

**Happy Learning! 🚀**

*Make sure your dataset is ready before running the SVM notebook!*

</div>
