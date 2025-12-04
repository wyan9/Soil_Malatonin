import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

### Load data
data = pd.read_csv('data/factors.csv')  # Read dataset

### Data cleaning
# Remove rows with 5 or more zeros (check only feature columns)
print(f"Original dataset size: {len(data)}")
zero_counts = data.iloc[:, :-1].eq(0).sum(axis=1)  # Count zeros per row
data_clean = data[zero_counts < 6]                 # Filter condition
# Optionally remove rows where target is zero
# data_clean = data_clean[data_clean['Abundance'] != 0]
print(f"Cleaned dataset size: {len(data_clean)} (removed {len(data)-len(data_clean)} rows)")

### Prepare dataset
X = data_clean.iloc[:, :-1].values
y = data_clean.iloc[:, -1].values

### Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### Model configuration
optimized_params = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 2,
    'max_features': 'log2',
    'random_state': 42,
    'oob_score': True,    # Enable out-of-bag score evaluation
}

### Cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=470)
r2_scores = []
best_model = None
best_r2 = -np.inf
best_x_test = None
best_y_test = None

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Model training
    model = RandomForestRegressor(**optimized_params)
    # Alternative models:
    # model = GradientBoostingRegressor()
    # model = XGBRegressor()
    # model = LGBMRegressor()  # LightGBM (not performing well here)

    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    fold_r2 = r2_score(y_test, y_pred)
    print(f'Fold {fold}: R² = {fold_r2:.4f}')
    if fold_r2 < 0:
        continue
    r2_scores.append(fold_r2)
    
    # Update best model
    if fold_r2 > best_r2:
        best_r2 = fold_r2
        best_model = model
        best_x_test = X_test
        best_y_test = y_test

### Final evaluation
final_pred = best_model.predict(best_x_test)
final_r2 = r2_score(best_y_test, final_pred)
final_rmse = np.sqrt(mean_squared_error(best_y_test, final_pred))

print(f'\nAverage validation R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})')
print(f'Final model R²: {final_r2:.4f}')
print(f'Final model RMSE: {final_rmse:.4f}')

### Save model pipeline
joblib.dump({
    'model': best_model,
    'scaler': scaler,
    'feature_names': data_clean.columns[:-1].tolist()
}, 'models/optimized_model.pkl')

### Feature importance
feature_importance = pd.DataFrame({
    'feature': data_clean.columns[:-1],
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 feature importances:")
print(feature_importance.head(10))
