# 🖼️ GPU-Accelerated SVM for Image Classification
# Optimized version for your Dog vs Cat classification

import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# GPU Libraries
try:
    import cupy as cp
    import cudf
    from cuml.svm import SVC as cuSVC
    from cuml.decomposition import PCA as cuPCA
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from thundersvm import SVC as ThunderSVC
    THUNDERSVM_AVAILABLE = True
except ImportError:
    THUNDERSVM_AVAILABLE = False

class GPUImageSVM:
    """GPU-accelerated SVM for image classification"""
    
    def __init__(self, method='cuml', kernel='rbf', C=1.0, gamma='scale', n_components=100):
        self.method = method
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.n_components = n_components
        self.model = None
        self.pca = None
        self.scaler = None
        
    def _prepare_gpu_data(self, X, y=None):
        """Convert data to GPU format"""
        if self.method == 'cuml':
            X_gpu = cudf.DataFrame(X) if not isinstance(X, cudf.DataFrame) else X
            if y is not None:
                y_gpu = cudf.Series(y) if not isinstance(y, cudf.Series) else y
                return X_gpu, y_gpu
            return X_gpu
        return X, y
    
    def fit(self, X_train, y_train, use_pca=True, verbose=True):
        """Train GPU SVM with optional PCA"""
        start_time = time.time()
        
        if verbose:
            print(f"\n🚀 Training GPU SVM ({self.method.upper()})...")
            print("="*50)
            print(f"Original data shape: {X_train.shape}")
        
        # Step 1: PCA for dimensionality reduction (if needed)
        if use_pca and X_train.shape[1] > self.n_components:
            if verbose:
                print(f"Applying PCA: {X_train.shape[1]} → {self.n_components} features")
            
            if self.method == 'cuml' and GPU_AVAILABLE:
                # GPU PCA
                X_train_gpu = cudf.DataFrame(X_train)
                self.pca = cuPCA(n_components=self.n_components)
                X_train_pca = self.pca.fit_transform(X_train_gpu)
            else:
                # CPU PCA fallback
                self.pca = PCA(n_components=self.n_components)
                X_train_pca = self.pca.fit_transform(X_train)
        else:
            X_train_pca = X_train
            
        # Step 2: Feature Scaling
        if verbose:
            print("Scaling features...")
            
        if self.method == 'cuml' and GPU_AVAILABLE:
            if not isinstance(X_train_pca, cudf.DataFrame):
                X_train_pca = cudf.DataFrame(X_train_pca)
            self.scaler = cuStandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_pca)
            y_train_gpu = cudf.Series(y_train)
        else:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_pca)
            y_train_gpu = y_train
        
        # Step 3: Train SVM
        if verbose:
            print(f"Training {self.kernel} SVM...")
            
        if self.method == 'cuml' and GPU_AVAILABLE:
            self.model = cuSVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma
            )
        elif self.method == 'thundersvm' and THUNDERSVM_AVAILABLE:
            self.model = ThunderSVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                gpu_id=0
            )
        else:
            # Fallback to CPU
            from sklearn.svm import SVC
            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma
            )
            
        self.model.fit(X_train_scaled, y_train_gpu)
        
        train_time = time.time() - start_time
        
        if verbose:
            print(f"✅ Training completed in {train_time:.2f} seconds")
            
        return train_time
    
    def predict(self, X_test):
        """Make predictions"""
        # Apply same preprocessing
        if self.pca is not None:
            if self.method == 'cuml' and GPU_AVAILABLE:
                X_test_gpu = cudf.DataFrame(X_test)
                X_test_pca = self.pca.transform(X_test_gpu)
            else:
                X_test_pca = self.pca.transform(X_test)
        else:
            X_test_pca = X_test
            
        if self.method == 'cuml' and GPU_AVAILABLE:
            if not isinstance(X_test_pca, cudf.DataFrame):
                X_test_pca = cudf.DataFrame(X_test_pca)
            X_test_scaled = self.scaler.transform(X_test_pca)
            y_pred = self.model.predict(X_test_scaled)
            # Convert back to CPU
            return y_pred.to_pandas().values if hasattr(y_pred, 'to_pandas') else y_pred
        else:
            X_test_scaled = self.scaler.transform(X_test_pca)
            return self.model.predict(X_test_scaled)
    
    def score(self, X_test, y_test):
        """Calculate accuracy"""
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

def compare_svm_methods(X_train, X_test, y_train, y_test):
    """Compare different SVM methods"""
    print("🎯 GPU SVM Method Comparison for Image Classification")
    print("="*70)
    
    methods = []
    
    # Add available methods
    if GPU_AVAILABLE:
        methods.append(('cuml', 'RAPIDS cuML'))
    if THUNDERSVM_AVAILABLE:
        methods.append(('thundersvm', 'ThunderSVM'))
    methods.append(('cpu', 'CPU Baseline'))
    
    results = {}
    
    for method_key, method_name in methods:
        print(f"\n🔄 Testing {method_name}...")
        
        try:
            # Initialize model
            svm = GPUImageSVM(
                method=method_key,
                kernel='rbf',
                C=1.0,
                gamma='scale',
                n_components=100  # Reduce dimensions for faster training
            )
            
            # Train
            train_time = svm.fit(X_train, y_train, use_pca=True, verbose=True)
            
            # Test
            start_test = time.time()
            accuracy = svm.score(X_test, y_test)
            test_time = time.time() - start_test
            
            results[method_name] = {
                'train_time': train_time,
                'test_time': test_time,
                'accuracy': accuracy
            }
            
            print(f"✅ {method_name}: {accuracy:.4f} accuracy in {train_time:.2f}s")
            
        except Exception as e:
            print(f"❌ {method_name} failed: {str(e)}")
    
    # Summary
    print("\n📊 FINAL COMPARISON")
    print("="*70)
    print(f"{'Method':<20} {'Train Time':<12} {'Test Time':<12} {'Accuracy':<12}")
    print("-"*70)
    
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['train_time']:<12.2f} {metrics['test_time']:<12.4f} {metrics['accuracy']:<12.4f}")

# Example usage for your image classification project
def optimize_image_svm():
    """
    Example of how to optimize your Dog vs Cat SVM classification
    Replace this with your actual image loading code
    """
    print("🖼️ Optimizing Image Classification SVM")
    print("="*50)
    
    # Simulate your image data (replace with actual loading)
    np.random.seed(42)
    n_samples = 1000
    n_features = 64 * 64 * 3  # Example: 64x64 RGB images
    
    X_train = np.random.randn(int(0.8 * n_samples), n_features)
    X_test = np.random.randn(int(0.2 * n_samples), n_features)
    y_train = np.random.randint(0, 2, int(0.8 * n_samples))  # Binary: dog=0, cat=1
    y_test = np.random.randint(0, 2, int(0.2 * n_samples))
    
    print(f"Simulated data: {X_train.shape[0]} training images, {X_test.shape[0]} test images")
    print(f"Image dimensions: {n_features} features per image")
    
    # Compare methods
    compare_svm_methods(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    # Check GPU availability
    print("🔍 Checking GPU Support...")
    print(f"RAPIDS cuML available: {GPU_AVAILABLE}")
    print(f"ThunderSVM available: {THUNDERSVM_AVAILABLE}")
    
    if not GPU_AVAILABLE and not THUNDERSVM_AVAILABLE:
        print("\n⚠️ No GPU SVM libraries found!")
        print("Install options:")
        print("1. RAPIDS cuML: conda install -c rapidsai -c nvidia -c conda-forge cuml")
        print("2. ThunderSVM: pip install thundersvm")
    
    # Run optimization example
    optimize_image_svm()
