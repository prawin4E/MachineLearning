# 🚀 GPU-Accelerated SVM Implementation
# This script shows how to use GPU for SVM training

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

# GPU Libraries
try:
    import cupy as cp
    import cudf
    from cuml.svm import SVC as cuSVC
    from cuml.decomposition import PCA as cuPCA
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
    print("✅ GPU libraries available!")
except ImportError:
    print("❌ GPU libraries not available. Install RAPIDS cuML:")
    print("conda install -c rapidsai -c nvidia -c conda-forge cuml")
    GPU_AVAILABLE = False

# Alternative: ThunderSVM
try:
    from thundersvm import SVC as ThunderSVC
    THUNDERSVM_AVAILABLE = True
    print("✅ ThunderSVM available!")
except ImportError:
    print("❌ ThunderSVM not available. Install with: pip install thundersvm")
    THUNDERSVM_AVAILABLE = False

def load_and_prepare_data():
    """Load and prepare the heart disease dataset"""
    # Load dataset
    df = pd.read_csv('../dataset/heart.csv')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def cpu_svm_baseline(X_train, X_test, y_train, y_test):
    """CPU SVM baseline for comparison"""
    print("\n🖥️  CPU SVM Baseline")
    print("="*50)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    from sklearn.svm import SVC
    start_time = time.time()
    
    svm_cpu = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_cpu.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = svm_cpu.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return train_time, accuracy

def gpu_svm_cuml(X_train, X_test, y_train, y_test):
    """GPU SVM using RAPIDS cuML"""
    if not GPU_AVAILABLE:
        print("❌ cuML not available")
        return None, None
    
    print("\n🚀 GPU SVM (RAPIDS cuML)")
    print("="*50)
    
    # Convert to GPU DataFrames
    X_train_gpu = cudf.DataFrame(X_train)
    X_test_gpu = cudf.DataFrame(X_test)
    y_train_gpu = cudf.Series(y_train.values)
    y_test_gpu = cudf.Series(y_test.values)
    
    # GPU Scaling
    scaler_gpu = cuStandardScaler()
    X_train_scaled_gpu = scaler_gpu.fit_transform(X_train_gpu)
    X_test_scaled_gpu = scaler_gpu.transform(X_test_gpu)
    
    # Train GPU SVM
    start_time = time.time()
    
    svm_gpu = cuSVC(kernel='rbf', C=1.0, gamma='scale')
    svm_gpu.fit(X_train_scaled_gpu, y_train_gpu)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred_gpu = svm_gpu.predict(X_test_scaled_gpu)
    
    # Convert back to CPU for accuracy calculation
    y_pred_cpu = y_pred_gpu.to_pandas().values
    y_test_cpu = y_test.values
    accuracy = accuracy_score(y_test_cpu, y_pred_cpu)
    
    print(f"Training time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return train_time, accuracy

def gpu_svm_thundersvm(X_train, X_test, y_train, y_test):
    """GPU SVM using ThunderSVM"""
    if not THUNDERSVM_AVAILABLE:
        print("❌ ThunderSVM not available")
        return None, None
    
    print("\n⚡ GPU SVM (ThunderSVM)")
    print("="*50)
    
    # Scale features (CPU)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ThunderSVM
    start_time = time.time()
    
    svm_thunder = ThunderSVC(
        kernel='rbf', 
        C=1.0, 
        gamma='scale',
        gpu_id=0  # Use first GPU
    )
    svm_thunder.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = svm_thunder.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return train_time, accuracy

def gpu_pca_svm_example(X_train, X_test, y_train, y_test):
    """Example with PCA + GPU SVM for high-dimensional data"""
    if not GPU_AVAILABLE:
        print("❌ cuML not available for PCA example")
        return
    
    print("\n🎯 GPU PCA + SVM (High-Dimensional Data)")
    print("="*50)
    
    # Simulate high-dimensional data (like your image features)
    # In practice, this would be your flattened image data
    np.random.seed(42)
    X_train_hd = np.random.randn(len(X_train), 1000)  # 1000 features
    X_test_hd = np.random.randn(len(X_test), 1000)
    
    # Convert to GPU
    X_train_gpu = cudf.DataFrame(X_train_hd)
    X_test_gpu = cudf.DataFrame(X_test_hd)
    y_train_gpu = cudf.Series(y_train.values)
    
    # GPU PCA
    print("Performing GPU PCA...")
    pca_gpu = cuPCA(n_components=50)
    X_train_pca_gpu = pca_gpu.fit_transform(X_train_gpu)
    X_test_pca_gpu = pca_gpu.transform(X_test_gpu)
    
    # GPU Scaling
    scaler_gpu = cuStandardScaler()
    X_train_scaled_gpu = scaler_gpu.fit_transform(X_train_pca_gpu)
    X_test_scaled_gpu = scaler_gpu.transform(X_test_pca_gpu)
    
    # GPU SVM
    print("Training GPU SVM...")
    start_time = time.time()
    
    svm_gpu = cuSVC(kernel='rbf', C=1.0, gamma='scale')
    svm_gpu.fit(X_train_scaled_gpu, y_train_gpu)
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred_gpu = svm_gpu.predict(X_test_scaled_gpu)
    y_pred_cpu = y_pred_gpu.to_pandas().values
    accuracy = accuracy_score(y_test.values, y_pred_cpu)
    
    print(f"PCA + SVM training time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

def main():
    """Main function to run all comparisons"""
    print("🎯 GPU-Accelerated SVM Comparison")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Results storage
    results = {}
    
    # CPU Baseline
    cpu_time, cpu_acc = cpu_svm_baseline(X_train, X_test, y_train, y_test)
    results['CPU SVM'] = {'time': cpu_time, 'accuracy': cpu_acc}
    
    # GPU cuML
    cuml_time, cuml_acc = gpu_svm_cuml(X_train, X_test, y_train, y_test)
    if cuml_time is not None:
        results['GPU cuML'] = {'time': cuml_time, 'accuracy': cuml_acc}
    
    # GPU ThunderSVM
    thunder_time, thunder_acc = gpu_svm_thundersvm(X_train, X_test, y_train, y_test)
    if thunder_time is not None:
        results['GPU ThunderSVM'] = {'time': thunder_time, 'accuracy': thunder_acc}
    
    # PCA + GPU example
    gpu_pca_svm_example(X_train, X_test, y_train, y_test)
    
    # Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Time (s)':<12} {'Accuracy':<12} {'Speedup':<10}")
    print("-"*60)
    
    baseline_time = results['CPU SVM']['time']
    for method, metrics in results.items():
        speedup = baseline_time / metrics['time'] if metrics['time'] > 0 else 0
        print(f"{method:<20} {metrics['time']:<12.4f} {metrics['accuracy']:<12.4f} {speedup:<10.2f}x")

if __name__ == "__main__":
    main()
