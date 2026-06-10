# ============================================================
# CODEALPHA INTERNSHIP — TASK 1: Iris Flower Classification
# Dataset: Auto-loaded from sklearn (no download needed!)
# Just run top to bottom!
# ============================================================

# ---------- STEP 1: Install Libraries ----------
# Run in terminal first:
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
print("✅ Libraries imported successfully!")

# ============================================================
# STEP 2: Load Dataset
# ============================================================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("✅ Iris dataset loaded!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Species: {df['species'].unique()}")

# ============================================================
# STEP 3: Explore the Data
# ============================================================
print("\n" + "="*55)
print("FIRST 5 ROWS:")
print("="*55)
print(df.head())

print("\n" + "="*55)
print("BASIC STATISTICS:")
print("="*55)
print(df.describe())

print("\n" + "="*55)
print("SPECIES COUNT:")
print("="*55)
print(df['species'].value_counts())

print("\n" + "="*55)
print("MISSING VALUES:")
print("="*55)
print(df.isnull().sum())

# ============================================================
# STEP 4: Visualizations
# ============================================================

# Chart 1: Species distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2ecc71', '#3498db', '#e74c3c']
species_counts = df['species'].value_counts()
axes[0].bar(species_counts.index, species_counts.values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
for i, (name, val) in enumerate(species_counts.items()):
    axes[0].text(i, val + 0.5, str(val), ha='center', fontweight='bold', fontsize=13)
axes[0].set_title('Species Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_ylim(0, 60)

# Chart 2: Sepal Length vs Sepal Width scatter
for i, species in enumerate(iris.target_names):
    subset = df[df['species'] == species]
    axes[1].scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                    label=species, color=colors[i], s=80, alpha=0.8, edgecolors='white')
axes[1].set_title('Sepal Length vs Sepal Width', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sepal Length (cm)')
axes[1].set_ylabel('Sepal Width (cm)')
axes[1].legend()
plt.suptitle('Iris Dataset — Exploratory Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_chart1_exploration.png', bbox_inches='tight')
plt.show()
print("✅ Chart 1 saved: iris_chart1_exploration.png")

# Chart 2: Feature distributions (box plots)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features = iris.feature_names
axes_flat = axes.flatten()
for i, feature in enumerate(features):
    data_by_species = [df[df['species']==sp][feature].values for sp in iris.target_names]
    bp = axes_flat[i].boxplot(data_by_species, patch_artist=True, labels=iris.target_names)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes_flat[i].set_title(feature.replace('(cm)', '').title(), fontweight='bold')
    axes_flat[i].set_ylabel('cm')
plt.suptitle('Feature Distribution by Species', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_chart2_boxplots.png', bbox_inches='tight')
plt.show()
print("✅ Chart 2 saved: iris_chart2_boxplots.png")

# Chart 3: Pairplot (all features vs all)
pair_colors = {'setosa': '#2ecc71', 'versicolor': '#3498db', 'virginica': '#e74c3c'}
fig, axes = plt.subplots(4, 4, figsize=(14, 12))
for i, feat1 in enumerate(features):
    for j, feat2 in enumerate(features):
        ax = axes[i][j]
        if i == j:
            for k, sp in enumerate(iris.target_names):
                subset = df[df['species']==sp]
                ax.hist(subset[feat1], alpha=0.6, color=colors[k], bins=15, edgecolor='white')
        else:
            for k, sp in enumerate(iris.target_names):
                subset = df[df['species']==sp]
                ax.scatter(subset[feat2], subset[feat1], color=colors[k], s=20, alpha=0.7)
        if i == 3: ax.set_xlabel(feat2.replace(' (cm)',''), fontsize=8)
        if j == 0: ax.set_ylabel(feat1.replace(' (cm)',''), fontsize=8)
        ax.tick_params(labelsize=7)
plt.suptitle('Iris — Feature Pair Plot', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('iris_chart3_pairplot.png', bbox_inches='tight')
plt.show()
print("✅ Chart 3 saved: iris_chart3_pairplot.png")

# ============================================================
# STEP 5: Prepare Data for Machine Learning
# ============================================================
X = df[iris.feature_names]
y = df['species']

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n✅ Data split: {len(X_train)} training | {len(X_test)} testing samples")

# ============================================================
# STEP 6: Train 3 Models & Compare
# ============================================================
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':        DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42)
}

results = {}
print("\n" + "="*55)
print("MODEL TRAINING & ACCURACY:")
print("="*55)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100
    results[name] = {'model': model, 'predictions': y_pred, 'accuracy': acc}
    print(f"  {name}: {acc:.1f}%")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_acc = results[best_model_name]['accuracy']
print(f"\n🏆 Best Model: {best_model_name} ({best_acc:.1f}% accuracy)")

# ============================================================
# STEP 7: Detailed Report for Best Model
# ============================================================
best_preds = results[best_model_name]['predictions']
print("\n" + "="*55)
print(f"CLASSIFICATION REPORT ({best_model_name}):")
print("="*55)
print(classification_report(y_test, best_preds))

# Chart 4: Accuracy comparison + Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy bar
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
short_names = ['KNN', 'Decision\nTree', 'SVM']
bar_colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = axes[0].bar(short_names, accuracies, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(80, 105)

# Confusion matrix
cm = confusion_matrix(y_test, best_preds, labels=iris.target_names)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=iris.target_names, yticklabels=iris.target_names,
            linewidths=0.5, cbar_kws={'label': 'Count'})
axes[1].set_title(f'Confusion Matrix\n({best_model_name})', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)

plt.suptitle('Iris Classification — Model Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_chart4_model_results.png', bbox_inches='tight')
plt.show()
print("✅ Chart 4 saved: iris_chart4_model_results.png")

# ============================================================
# STEP 8: Predict a New Flower
# ============================================================
print("\n" + "="*55)
print("PREDICT A NEW FLOWER:")
print("="*55)
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # typical setosa
new_scaled = scaler.transform(new_flower)
best_clf = results[best_model_name]['model']
prediction = best_clf.predict(new_scaled)
print(f"  Input: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
print(f"  Predicted Species: {prediction[0].upper()}")

print("\n" + "="*55)
print("🎉 IRIS CLASSIFICATION COMPLETE!")
print("="*55)
print(f"  Best model: {best_model_name}")
print(f"  Best accuracy: {best_acc:.1f}%")
print("  Charts saved: 4 PNG files")
print("="*55)
