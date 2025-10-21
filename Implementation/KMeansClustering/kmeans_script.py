#!/usr/bin/env python
# coding: utf-8

# # 🎯 K-Means Clustering with Elbow Method
# 
# ## 📊 Complete Implementation with Customer Segmentation

# This comprehensive script demonstrates K-Means Clustering with the Elbow Method
# Copy sections to Jupyter notebook cells as needed

# ## Cell 1: Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples
)

from scipy import stats
from scipy.spatial.distance import cdist
import warnings
import pickle
import time

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

print("✅ Libraries imported successfully!")


# ## Cell 2: Load/Create Dataset

# Try to load dataset or create synthetic data
try:
    df = pd.read_csv('../../dataset/Mall_Customers.csv')
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("⚠️ Dataset not found. Creating synthetic mall customers data...")
    
    np.random.seed(42)
    n_samples = 200
    
    # Create 5 distinct clusters
    cluster1 = np.random.multivariate_normal([25, 30], [[20, 0], [0, 20]], 40)
    cluster2 = np.random.multivariate_normal([75, 75], [[25, 0], [0, 25]], 40)
    cluster3 = np.random.multivariate_normal([25, 75], [[20, 0], [0, 20]], 40)
    cluster4 = np.random.multivariate_normal([75, 30], [[25, 0], [0, 25]], 40)
    cluster5 = np.random.multivariate_normal([50, 50], [[15, 0], [0, 15]], 40)
    
    income_spending = np.vstack([cluster1, cluster2, cluster3, cluster4, cluster5])
    
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 71, n_samples),
        'Annual Income (k$)': np.clip(income_spending[:, 0], 15, 137).astype(int),
        'Spending Score (1-100)': np.clip(income_spending[:, 1], 1, 99).astype(int)
    }
    
    df = pd.DataFrame(data)
    print("✅ Synthetic dataset created successfully!")

print(f"\nDataset shape: {df.shape}")
print(df.head())


# ## Cell 3: Exploratory Data Analysis (EDA)

print("\n📊 Dataset Information:")
print("="*70)
print(df.info())

print("\n📈 Statistical Summary:")
print("="*70)
print(df.describe())

# Check missing values
print("\n🔍 Missing Values:")
print(df.isnull().sum())

# Check duplicates
duplicates = df.duplicated().sum()
print(f"\n🔍 Duplicate rows: {duplicates}")


# ## Cell 4: Visualize Feature Distributions

numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors_hist = ['skyblue', 'lightcoral', 'lightgreen']

for idx, col in enumerate(numerical_features):
    axes[idx].hist(df[col], bins=20, edgecolor='black', alpha=0.7, color=colors_hist[idx])
    axes[idx].set_title(f'Distribution of {col}', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel(col, fontsize=10)
    axes[idx].set_ylabel('Frequency', fontsize=10)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Cell 5: Correlation Analysis

plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, linewidths=1, square=True, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Cell 6: Income vs Spending Scatter Plot

plt.figure(figsize=(12, 8))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
            alpha=0.6, s=100, c='steelblue', edgecolors='black')
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.title('Annual Income vs Spending Score', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('income_vs_spending.png', dpi=150, bbox_inches='tight')
plt.show()

print("💡 Visual inspection suggests possible natural groupings in the data!")


# ## Cell 7: Feature Selection and Scaling

# Select features for clustering
feature_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[feature_cols].values

print(f"✅ Features selected: {feature_cols}")
print(f"Feature matrix shape: {X.shape}")

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✅ Feature scaling completed!")
print(f"Mean of scaled features: {X_scaled.mean(axis=0).round(10)}")
print(f"Std of scaled features: {X_scaled.std(axis=0).round(2)}")


# ## Cell 8: Elbow Method - Calculate WCSS

print("\n🔍 Calculating WCSS for K = 1 to 10...")
print("="*70)

K_range = range(1, 11)
wcss = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    print(f"K={k:2d} | WCSS (Inertia) = {kmeans.inertia_:.2f}")

print("\n✅ WCSS calculation completed!")


# ## Cell 9: Plot Elbow Curve

plt.figure(figsize=(12, 7))
plt.plot(K_range, wcss, marker='o', linewidth=2, markersize=10, color='steelblue', 
         markerfacecolor='red', markeredgecolor='black')
plt.xlabel('Number of Clusters (K)', fontsize=13)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=13)
plt.title('Elbow Method For Optimal K', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n💡 Look for the 'elbow point' where the curve bends sharply!")


# ## Cell 10: Silhouette Score Analysis

print("\n🔍 Calculating Silhouette Scores for K = 2 to 10...")
print("="*70)

K_range_silhouette = range(2, 11)
silhouette_scores = []

for k in K_range_silhouette:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"K={k:2d} | Silhouette Score = {silhouette_avg:.4f}")

print("\n✅ Silhouette Score calculation completed!")


# ## Cell 11: Plot Silhouette Scores

plt.figure(figsize=(12, 7))
plt.plot(K_range_silhouette, silhouette_scores, marker='s', linewidth=2, markersize=10, 
         color='green', markerfacecolor='yellow', markeredgecolor='black')
plt.xlabel('Number of Clusters (K)', fontsize=13)
plt.ylabel('Silhouette Score', fontsize=13)
plt.title('Silhouette Score vs Number of Clusters', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(K_range_silhouette)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('silhouette_scores.png', dpi=150, bbox_inches='tight')
plt.show()

optimal_k_silhouette = K_range_silhouette[np.argmax(silhouette_scores)]
print(f"\n🎯 Optimal K (Silhouette Score): {optimal_k_silhouette}")


# ## Cell 12: Comprehensive Evaluation Metrics

print("\n📊 Comprehensive Cluster Evaluation Metrics")
print("="*100)
print(f"{'K':<5} {'WCSS':<15} {'Silhouette':<15} {'Davies-Bouldin':<20} {'Calinski-Harabasz':<20}")
print("="*100)

davies_bouldin_scores = []
calinski_harabasz_scores = []

for i, k in enumerate(K_range_silhouette):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    sil_score = silhouette_scores[i]
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    
    davies_bouldin_scores.append(db_score)
    calinski_harabasz_scores.append(ch_score)
    
    wcss_value = wcss[k-1]
    
    print(f"{k:<5} {wcss_value:<15.2f} {sil_score:<15.4f} {db_score:<20.4f} {ch_score:<20.2f}")

print("="*100)


# ## Cell 13: Plot All Metrics

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# WCSS
axes[0, 0].plot(K_range, wcss, marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[0, 0].set_ylabel('WCSS (Inertia)', fontsize=11)
axes[0, 0].set_title('Elbow Method (WCSS)', fontsize=13, fontweight='bold')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xticks(K_range)

# Silhouette Score
axes[0, 1].plot(K_range_silhouette, silhouette_scores, marker='s', linewidth=2, markersize=8, color='green')
axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[0, 1].set_ylabel('Silhouette Score', fontsize=11)
axes[0, 1].set_title('Silhouette Score', fontsize=13, fontweight='bold')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xticks(K_range_silhouette)

# Davies-Bouldin Index
axes[1, 0].plot(K_range_silhouette, davies_bouldin_scores, marker='^', linewidth=2, markersize=8, color='red')
axes[1, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[1, 0].set_ylabel('Davies-Bouldin Index', fontsize=11)
axes[1, 0].set_title('Davies-Bouldin Index', fontsize=13, fontweight='bold')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xticks(K_range_silhouette)

# Calinski-Harabasz Score
axes[1, 1].plot(K_range_silhouette, calinski_harabasz_scores, marker='D', linewidth=2, markersize=8, color='purple')
axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[1, 1].set_ylabel('Calinski-Harabasz Score', fontsize=11)
axes[1, 1].set_title('Calinski-Harabasz Score', fontsize=13, fontweight='bold')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xticks(K_range_silhouette)

plt.tight_layout()
plt.savefig('all_metrics.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Cell 14: Determine Optimal K

print("\n🎯 Optimal K Determination")
print("="*70)

optimal_k_davies = K_range_silhouette[np.argmin(davies_bouldin_scores)]
optimal_k_calinski = K_range_silhouette[np.argmax(calinski_harabasz_scores)]

print(f"Based on Silhouette Score: K = {optimal_k_silhouette}")
print(f"Based on Davies-Bouldin Index: K = {optimal_k_davies}")
print(f"Based on Calinski-Harabasz Score: K = {optimal_k_calinski}")

# Choose final optimal K
optimal_k = 5

print(f"\n✅ Final Optimal K Selected: {optimal_k}")
print("\n💡 Business Justification:")
print("   5 clusters represent distinct customer segments")


# ## Cell 15: Train Final K-Means Model

print(f"\n🤖 Training K-Means with K={optimal_k}...")
print("="*70)

kmeans_final = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)

cluster_labels = kmeans_final.fit_predict(X_scaled)
cluster_centers = kmeans_final.cluster_centers_

df['Cluster'] = cluster_labels

print(f"✅ K-Means clustering completed!")
print(f"\nModel Parameters:")
print(f"  Number of clusters: {kmeans_final.n_clusters}")
print(f"  Iterations run: {kmeans_final.n_iter_}")
print(f"  Final inertia: {kmeans_final.inertia_:.2f}")


# ## Cell 16: Cluster Distribution Analysis

print("\n📊 Cluster Distribution:")
print("="*70)
cluster_counts = df['Cluster'].value_counts().sort_index()
print(cluster_counts)
print(f"\nPercentage:")
print((cluster_counts / len(df) * 100).round(2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
cluster_counts.plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Customer Count per Cluster', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Cluster', fontsize=12)
axes[0].set_ylabel('Number of Customers', fontsize=12)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].grid(alpha=0.3, axis='y')

# Pie chart
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
axes[1].pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index], 
            autopct='%1.1f%%', colors=colors_pie, startangle=90)
axes[1].set_title('Cluster Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('cluster_distribution.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Cell 17: Cluster Profiling

print("\n📋 Cluster Profile - Mean Values:")
print("="*100)
cluster_profile = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_profile.round(2))

print("\n📋 Detailed Cluster Profiles:")
print("="*100)
for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"  Avg Age: {cluster_data['Age'].mean():.1f} years")
    print(f"  Avg Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
    print(f"  Avg Spending: {cluster_data['Spending Score (1-100)'].mean():.1f}/100")


# ## Cell 18: Visualize Cluster Profiles

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, feature in enumerate(features):
    cluster_profile[feature].plot(kind='bar', ax=axes[idx], color=colors[idx], edgecolor='black')
    axes[idx].set_title(f'Average {feature} by Cluster', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Cluster', fontsize=10)
    axes[idx].set_ylabel(f'Average {feature}', fontsize=10)
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=0)
    axes[idx].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cluster_profiles.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Cell 19: 2D Cluster Visualization

plt.figure(figsize=(14, 10))

colors_clusters = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], 
                cluster_data['Spending Score (1-100)'],
                s=100, c=colors_clusters[i], label=f'Cluster {i}',
                alpha=0.6, edgecolors='black', linewidth=0.5)

# Plot centroids
centroids_original = scaler.inverse_transform(cluster_centers)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1],
            s=400, c='yellow', marker='*', edgecolors='black', linewidth=2,
            label='Centroids', zorder=10)

plt.xlabel('Annual Income (k$)', fontsize=13)
plt.ylabel('Spending Score (1-100)', fontsize=13)
plt.title('Customer Segments - K-Means Clustering', fontsize=16, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n🎨 Cluster Visualization Complete!")
print("⭐ Yellow stars represent cluster centroids")


# ## Cell 20: 3D Visualization

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    ax.scatter(cluster_data['Annual Income (k$)'],
               cluster_data['Spending Score (1-100)'],
               cluster_data['Age'],
               s=100, c=colors_clusters[i], label=f'Cluster {i}',
               alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Annual Income (k$)', fontsize=11)
ax.set_ylabel('Spending Score (1-100)', fontsize=11)
ax.set_zlabel('Age', fontsize=11)
ax.set_title('3D Customer Segmentation', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('cluster_3d.png', dpi=150, bbox_inches='tight')
plt.show()


# ## Cell 21: Final Model Evaluation

final_inertia = kmeans_final.inertia_
final_silhouette = silhouette_score(X_scaled, cluster_labels)
final_davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
final_calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)

print("\n📊 FINAL MODEL EVALUATION METRICS")
print("="*70)
print(f"Number of Clusters (K): {optimal_k}")
print(f"\n1. WCSS (Inertia): {final_inertia:.4f}")
print(f"2. Silhouette Score: {final_silhouette:.4f}")
print(f"3. Davies-Bouldin Index: {final_davies_bouldin:.4f}")
print(f"4. Calinski-Harabasz Score: {final_calinski_harabasz:.4f}")
print("="*70)


# ## Cell 22: Business Interpretation

print("\n🏷️ CLUSTER BUSINESS INTERPRETATION")
print("="*100)

cluster_names = {}
marketing_strategies = {}

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()
    
    if avg_income < 40 and avg_spending < 40:
        name = "Budget Conscious"
        strategy = "Discounts, value deals, loyalty rewards"
    elif avg_income < 40 and avg_spending >= 60:
        name = "Aspirational Shoppers"
        strategy = "Installment plans, credit options, seasonal promotions"
    elif avg_income >= 70 and avg_spending < 40:
        name = "Potential High Value"
        strategy = "Premium products, exclusive previews, personalization"
    elif avg_income >= 70 and avg_spending >= 60:
        name = "Premium Customers"
        strategy = "VIP treatment, exclusive events, premium launches"
    else:
        name = "Average Customers"
        strategy = "Standard marketing, balanced mix, regular promotions"
    
    cluster_names[i] = name
    marketing_strategies[i] = strategy
    
    print(f"\nCluster {i}: {name}")
    print(f"{'─'*95}")
    print(f"  📊 Size: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"  💰 Avg Income: ${avg_income:.1f}k")
    print(f"  🛒 Avg Spending: {avg_spending:.1f}/100")
    print(f"  🎯 Strategy: {strategy}")

df['Cluster_Name'] = df['Cluster'].map(cluster_names)


# ## Cell 23: Save Model

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_final, f)

with open('kmeans_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('cluster_names.pkl', 'wb') as f:
    pickle.dump(cluster_names, f)

print("\n✅ Model, scaler, and cluster names saved successfully!")
print("Files: kmeans_model.pkl, kmeans_scaler.pkl, cluster_names.pkl")


# ## Cell 24: Predict for New Customers

print("\n🔮 Predicting Clusters for New Customers")
print("="*70)

new_customers = np.array([
    [25, 30],  # Low income, low spending
    [80, 85],  # High income, high spending
    [50, 50],  # Medium income, medium spending
])

new_customers_scaled = scaler.transform(new_customers)
new_clusters = kmeans_final.predict(new_customers_scaled)

for i, (customer, cluster) in enumerate(zip(new_customers, new_clusters)):
    print(f"\nNew Customer {i+1}:")
    print(f"  Income: ${customer[0]}k, Spending: {customer[1]}/100")
    print(f"  → Cluster {cluster}: {cluster_names[cluster]}")
    print(f"  → Strategy: {marketing_strategies[cluster]}")


print("\n" + "="*70)
print("✅ K-MEANS CLUSTERING PROJECT COMPLETE!")
print("="*70)
print("\n📚 Key Achievements:")
print("  ✓ Implemented K-Means with Elbow Method")
print("  ✓ Evaluated with multiple metrics")
print("  ✓ Visualized clusters in 2D and 3D")
print("  ✓ Profiled customer segments")
print("  ✓ Provided business insights")
print("  ✓ Saved model for production use")
print("\n🎯 Happy Clustering! 🚀")



