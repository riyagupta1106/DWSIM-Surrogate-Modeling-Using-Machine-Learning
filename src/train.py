import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# 1. Load Dataset
data_path = os.path.join("data", "dataset.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Generate it first!")

df = pd.read_csv(data_path)
print(f"Dataset loaded successfully with shape: {df.shape}")

# 2. Define Features (X) and Targets (y)
X = df[["Feed_T", "Feed_P", "Feed_Flow", "Feed_Benzene", "Feed_Toluene"]]
y = df[[
    "Dist_T", "Dist_P", "Dist_Flow", "Dist_Benzene", "Dist_Toluene",
    "Bottom_T", "Bottom_P", "Bottom_Flow", "Bottom_Benzene", "Bottom_Toluene"
]]

# 3. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Hyperparameter Optimization via RandomizedSearchCV (Fixed grid)
print("\n--- Running Hyperparameter Optimization ---")
base_model = RandomForestRegressor(random_state=42)

# Corrected parameter grid without the obsolete 'auto' keyword
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 1.0]
}

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=15,            # Try 15 combinations
    cv=3,                 # 3-fold cross validation
    scoring='r2',
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)

random_search.fit(X_train_scaled, y_train)

best_model = random_search.best_estimator_
print(f"\nOptimal Hyperparameters Found:")
for param, value in random_search.best_params_.items():
    print(f"  -> {param}: {value}")

# 6. Evaluate Optimized Model on Test Set
y_pred = best_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- Optimized Model Performance ---")
print(f"  -> Average R² Score: {r2:.4f}")
print(f"  -> Root Mean Squared Error (RMSE): {rmse:.4f}")

# 7. Save Optimized Artifacts Cleanly
os.makedirs("models", exist_ok=True)
model_save_path = os.path.join("models", "best_surrogate_model.pkl")
scaler_save_path = os.path.join("models", "scaler.pkl")

joblib.dump(best_model, model_save_path)
joblib.dump(scaler, scaler_save_path)

print(f"\nOptimized artifacts successfully saved and verified!")