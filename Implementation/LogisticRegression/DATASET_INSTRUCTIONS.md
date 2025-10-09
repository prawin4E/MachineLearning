# 📥 Dataset Download Instructions

## Heart Disease UCI Dataset

### Option 1: Kaggle (Recommended)

1. Visit [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
2. Click "Download" button (you may need to sign in to Kaggle)
3. Extract the downloaded ZIP file
4. Locate the `heart.csv` file
5. Move it to: `MachineLearning/dataset/heart.csv`

### Option 2: UCI Machine Learning Repository

1. Visit [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
2. Download the processed Cleveland data
3. Save as `heart.csv`
4. Move it to: `MachineLearning/dataset/heart.csv`

### Option 3: Direct Download Links

You can also find the dataset on these platforms:
- [GitHub - Heart Disease Dataset](https://github.com/rashakil-ds/Heart-Disease-Prediction-Dataset)
- Search "heart disease UCI dataset" on Google

### Required File Location

```
MachineLearning/
└── dataset/
    ├── CAR DETAILS FROM CAR DEKHO.csv
    └── heart.csv  ← Place the downloaded file here
```

### Dataset Details

- **Name**: heart.csv
- **Size**: ~10 KB
- **Records**: ~300 rows
- **Features**: 14 columns (13 features + 1 target)
- **Format**: CSV (Comma-separated values)

### Verification

After downloading, verify the file:

```bash
cd MachineLearning/dataset
ls -lh heart.csv
```

You should see a file around 10-15 KB in size.

### Alternative: Generate Sample Data

If you cannot download the dataset, you can use this Python code to generate sample data:

```python
import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 300

data = {
    'age': np.random.randint(29, 78, n_samples),
    'sex': np.random.choice([0, 1], n_samples),
    'cp': np.random.choice([0, 1, 2, 3], n_samples),
    'trestbps': np.random.randint(94, 201, n_samples),
    'chol': np.random.randint(126, 565, n_samples),
    'fbs': np.random.choice([0, 1], n_samples),
    'restecg': np.random.choice([0, 1, 2], n_samples),
    'thalach': np.random.randint(71, 203, n_samples),
    'exang': np.random.choice([0, 1], n_samples),
    'oldpeak': np.random.uniform(0, 6.2, n_samples).round(1),
    'slope': np.random.choice([0, 1, 2], n_samples),
    'ca': np.random.choice([0, 1, 2, 3], n_samples),
    'thal': np.random.choice([1, 2, 3], n_samples),
    'target': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)
df.to_csv('../../dataset/heart.csv', index=False)
print("Sample dataset created successfully!")
```

**Note**: Sample data is for learning purposes only and won't produce accurate medical predictions.

---

## Need Help?

If you encounter issues downloading the dataset:
1. Check the Kaggle link is still active
2. Ensure you're signed in to Kaggle
3. Try alternative sources listed above
4. Use the sample data generator as a last resort

---

**Ready to Start?** Once you have the dataset in place, open `LogisticRegression.ipynb` and run the cells!
