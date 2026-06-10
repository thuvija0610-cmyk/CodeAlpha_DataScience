# ============================================================
# CODEALPHA INTERNSHIP — TASK 3: Car Price Prediction
# Dataset: Built-in (no download needed!)
# Uses Machine Learning to predict car prices
# ============================================================

# ---------- Install Libraries ----------
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 120
print("✅ Libraries loaded!")

# ============================================================
# STEP 2: Create Car Dataset
# ============================================================
np.random.seed(42)
n = 200

brands      = ['Toyota','Honda','BMW','Mercedes','Hyundai','Ford','Audi','Kia','Maruti','Tata']
brand_base  = {'Toyota':800,'Honda':750,'BMW':2500,'Mercedes':3000,'Hyundai':650,
               'Ford':700,'Audi':2200,'Kia':600,'Maruti':400,'Tata':450}
fuel_types  = ['Petrol','Diesel','Electric','CNG']
transmissions = ['Manual','Automatic']

brand_list  = np.random.choice(brands, n)
year_list   = np.random.randint(2010, 2024, n)
km_list     = np.random.randint(5000, 150000, n)
fuel_list   = np.random.choice(fuel_types, n, p=[0.45,0.30,0.10,0.15])
trans_list  = np.random.choice(transmissions, n, p=[0.55,0.45])
hp_list     = np.random.randint(70, 400, n)
seats_list  = np.random.choice([5, 7, 8], n, p=[0.7, 0.25, 0.05])
owners_list = np.random.choice([1, 2, 3, 4], n, p=[0.5, 0.3, 0.15, 0.05])

# Calculate price based on features
prices = []
for i in range(n):
    base   = brand_base[brand_list[i]]
    age    = 2024 - year_list[i]
    depr   = base * (0.85 ** age)
    km_pen = (km_list[i] / 10000) * 15
    hp_bon = (hp_list[i] - 100) * 2
    fuel_m = {'Petrol':1.0,'Diesel':1.1,'Electric':1.3,'CNG':0.9}[fuel_list[i]]
    trans_m= 1.08 if trans_list[i]=='Automatic' else 1.0
    owner_p= {1:1.0,2:0.92,3:0.85,4:0.75}[owners_list[i]]
    price  = max(100, (depr - km_pen + hp_bon) * fuel_m * trans_m * owner_p)
    noise  = np.random.uniform(0.9, 1.1)
    prices.append(round(price * noise, 2))

df = pd.DataFrame({
    'Brand': brand_list,
    'Year': year_list,
    'Kilometers_Driven': km_list,
    'Fuel_Type': fuel_list,
    'Transmission': trans_list,
    'Horsepower': hp_list,
    'Seats': seats_list,
    'Owner_Count': owners_list,
    'Price_Thousands': prices
})

df['Car_Age'] = 2024 - df['Year']
print("✅ Car dataset created!")
print(f"   Shape: {df.shape}")
print(f"   Price range: ₹{df['Price_Thousands'].min():.0f}K – ₹{df['Price_Thousands'].max():.0f}K")

# ============================================================
# STEP 3: Explore the Data
# ============================================================
print("\n" + "="*55)
print("DATASET INFO:")
print("="*55)
print(df.head())

print("\n" + "="*55)
print("PRICE STATISTICS:")
print("="*55)
print(f"  Mean price:   ₹{df['Price_Thousands'].mean():.0f}K")
print(f"  Median price: ₹{df['Price_Thousands'].median():.0f}K")
print(f"  Min price:    ₹{df['Price_Thousands'].min():.0f}K")
print(f"  Max price:    ₹{df['Price_Thousands'].max():.0f}K")

print("\n" + "="*55)
print("AVERAGE PRICE BY BRAND:")
print("="*55)
brand_avg = df.groupby('Brand')['Price_Thousands'].mean().sort_values(ascending=False)
for brand, price in brand_avg.items():
    print(f"  {brand}: ₹{price:.0f}K")

# ============================================================
# STEP 4: Visualizations
# ============================================================

# Chart 1: Price distribution + Price by brand
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['Price_Thousands'], bins=30, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].axvline(df['Price_Thousands'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: ₹{df['Price_Thousands'].mean():.0f}K")
axes[0].set_title('Car Price Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Price (₹ Thousands)')
axes[0].set_ylabel('Count')
axes[0].legend()

brand_avg_sorted = df.groupby('Brand')['Price_Thousands'].mean().sort_values(ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(brand_avg_sorted)))
axes[1].barh(brand_avg_sorted.index, brand_avg_sorted.values, color=colors, edgecolor='white')
axes[1].set_title('Average Price by Brand', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Price (₹ Thousands)')
plt.suptitle('Car Price Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('car_chart1_price_dist.png', bbox_inches='tight')
plt.show()
print("✅ Chart 1 saved: car_chart1_price_dist.png")

# Chart 2: Key feature relationships
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].scatter(df['Horsepower'], df['Price_Thousands'], alpha=0.6, color='#e74c3c', s=40)
axes[0,0].set_title('Horsepower vs Price', fontweight='bold')
axes[0,0].set_xlabel('Horsepower')
axes[0,0].set_ylabel('Price (₹K)')

axes[0,1].scatter(df['Kilometers_Driven'], df['Price_Thousands'], alpha=0.6, color='#3498db', s=40)
axes[0,1].set_title('Km Driven vs Price', fontweight='bold')
axes[0,1].set_xlabel('Kilometers Driven')
axes[0,1].set_ylabel('Price (₹K)')

fuel_price = df.groupby('Fuel_Type')['Price_Thousands'].mean().sort_values(ascending=False)
axes[1,0].bar(fuel_price.index, fuel_price.values,
              color=['#e74c3c','#3498db','#2ecc71','#f39c12'][:len(fuel_price)],
              edgecolor='white', width=0.5)
axes[1,0].set_title('Average Price by Fuel Type', fontweight='bold')
axes[1,0].set_ylabel('Price (₹K)')

trans_price = df.groupby('Transmission')['Price_Thousands'].mean()
axes[1,1].bar(trans_price.index, trans_price.values,
              color=['#9b59b6','#e67e22'], edgecolor='white', width=0.4)
axes[1,1].set_title('Average Price by Transmission', fontweight='bold')
axes[1,1].set_ylabel('Price (₹K)')

plt.suptitle('Feature vs Price Relationships', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('car_chart2_features.png', bbox_inches='tight')
plt.show()
print("✅ Chart 2 saved: car_chart2_features.png")

# Chart 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            mask=mask, ax=ax, linewidths=0.5,
            annot_kws={'size': 10, 'weight': 'bold'})
ax.set_title('Feature Correlation Heatmap', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('car_chart3_correlation.png', bbox_inches='tight')
plt.show()
print("✅ Chart 3 saved: car_chart3_correlation.png")

# ============================================================
# STEP 5: Prepare Data for ML
# ============================================================
le = LabelEncoder()
df_ml = df.copy()
for col in ['Brand', 'Fuel_Type', 'Transmission']:
    df_ml[col] = le.fit_transform(df_ml[col])

features = ['Brand','Car_Age','Kilometers_Driven','Fuel_Type',
            'Transmission','Horsepower','Seats','Owner_Count']
X = df_ml[features]
y = df_ml['Price_Thousands']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"\n✅ Data split: {len(X_train)} train | {len(X_test)} test samples")

# ============================================================
# STEP 6: Train 3 Models
# ============================================================
models = {
    'Linear Regression':       LinearRegression(),
    'Random Forest':           RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting':       GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
print("\n" + "="*55)
print("MODEL RESULTS:")
print("="*55)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    results[name] = {'model': model, 'preds': preds, 'mae': mae, 'rmse': rmse, 'r2': r2}
    print(f"  {name}:")
    print(f"    R² Score: {r2:.4f} | MAE: ₹{mae:.0f}K | RMSE: ₹{rmse:.0f}K")

best_name = max(results, key=lambda x: results[x]['r2'])
print(f"\n🏆 Best Model: {best_name} (R² = {results[best_name]['r2']:.4f})")

# ============================================================
# STEP 7: Model Performance Charts
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² comparison
r2_vals = [results[m]['r2'] for m in results]
model_labels = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
bar_colors = ['#3498db','#2ecc71','#e74c3c']
bars = axes[0].bar(model_labels, r2_vals, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.5)
for bar, val in zip(bars, r2_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontweight='bold', fontsize=12)
axes[0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('R² Score (higher = better)')
axes[0].set_ylim(0, 1.1)

# Actual vs Predicted
best_preds = results[best_name]['preds']
axes[1].scatter(y_test, best_preds, alpha=0.6, color='#9b59b6', s=50, edgecolors='white')
min_val, max_val = min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
axes[1].set_title(f'Actual vs Predicted\n({best_name})', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Actual Price (₹K)')
axes[1].set_ylabel('Predicted Price (₹K)')
axes[1].legend()

plt.suptitle('Car Price Prediction — Model Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('car_chart4_model_results.png', bbox_inches='tight')
plt.show()
print("✅ Chart 4 saved: car_chart4_model_results.png")

# Chart 5: Feature Importance (Random Forest)
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Feature Importance (Random Forest)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('car_chart5_feature_importance.png', bbox_inches='tight')
plt.show()
print("✅ Chart 5 saved: car_chart5_feature_importance.png")

# ============================================================
# STEP 8: Predict Price of a New Car
# ============================================================
print("\n" + "="*55)
print("PREDICT PRICE OF A NEW CAR:")
print("="*55)
# Toyota, 3 years old, 30000 km, Petrol, Manual, 120hp, 5 seats, 1 owner
new_car = pd.DataFrame([[
    list(brand_base.keys()).index('Toyota'),  # Brand encoded
    3,       # Car_Age
    30000,   # Kilometers_Driven
    2,       # Fuel_Type (Petrol encoded)
    0,       # Transmission (Manual encoded)
    120,     # Horsepower
    5,       # Seats
    1        # Owner_Count
]], columns=features)
new_car_scaled = scaler.transform(new_car)
best_model = results[best_name]['model']
predicted_price = best_model.predict(new_car_scaled)[0]
print(f"  Car: Toyota | 3 years old | 30,000 km | Petrol | Manual | 120 HP")
print(f"  Predicted Price: ₹{predicted_price:.0f},000")

df.to_csv('car_price_results.csv', index=False)
print("\n✅ Results saved: car_price_results.csv")

print("\n" + "="*55)
print("🎉 CAR PRICE PREDICTION COMPLETE!")
print("="*55)
print(f"  Best model: {best_name}")
print(f"  R² Score:   {results[best_name]['r2']:.4f}")
print(f"  MAE:        ₹{results[best_name]['mae']:.0f}K")
print("  Charts saved: 5 PNG files")
print("="*55)
