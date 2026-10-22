import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.shipment_prep import FEATURES
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

print("Starting Part B: Regression Modeling...")

# Load the clean modeling table
df = pd.read_csv("data/modelling_table.csv")

# Define our inputs (X) and the target we want to predict (y)
X = df[FEATURES]
y = df["delivery_hours"]

# --- Task 1: Split into training, validation, and test ---
# First, split off 70% for training, leaving a temporary 30%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Second, split that temporary 30% strictly in half (50% of 30% = 15% each)
# This gives us our validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Data Split: {len(X_train)} Train | {len(X_val)} Validation | {len(X_test)} Test")

# --- Task 2: Scaling without leakage ---
# Initialize the scaler
scaler = StandardScaler()

# FIT the scaler only on the training data to learn the mean and variance.
# Then TRANSFORM the training data.
X_train_scaled = scaler.fit_transform(X_train)

# ONLY TRANSFORM the validation and test sets using the ruler learned from X_train.
# Do NOT use fit_transform here!
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# To prove we avoided leakage, the training scaled columns should have a mean exactly at 0.
# The test set scaled columns will have a mean slightly off 0, proving they didn't influence the scaler.
train_scaled_max_abs_mean = np.abs(X_train_scaled.mean(axis=0)).max()
test_scaled_max_abs_mean = np.abs(X_test_scaled.mean(axis=0)).max()

# Write our checks to a CSV
checks = pd.DataFrame({
    "metric": [
        "train_rows", "val_rows", "test_rows", 
        "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"
    ],
    "value": [
        len(X_train), len(X_val), len(X_test), 
        train_scaled_max_abs_mean, test_scaled_max_abs_mean
    ]
})

# Round float values slightly for clean output, though keeping them highly precise for the check
checks["value"] = checks["value"].apply(lambda x: round(x, 10) if isinstance(x, float) else x)
checks.to_csv("outputs/part_b_checks.csv", index=False)

print("Tasks 1 & 2 complete! part_b_checks.csv generated.")


# --- Task 3: The Baseline ---
print("Calculating Baseline...")
# 1. Calculate the mean delivery hours from the training data
baseline_prediction = y_train.mean()

# 2. Create an array predicting this exact same number for every parcel in the validation set
y_val_pred_baseline = np.full_like(y_val, fill_value=baseline_prediction)

# 3. Calculate how wrong the baseline was
base_mae = mean_absolute_error(y_val, y_val_pred_baseline)
base_mse = mean_squared_error(y_val, y_val_pred_baseline)
base_rmse = np.sqrt(base_mse)
base_r2 = r2_score(y_val, y_val_pred_baseline)

# We will save this dictionary later in Task 5
metrics_list = []
metrics_list.append({
    "model": "mean_baseline", "split": "val",
    "MAE": base_mae, "MSE": base_mse, "RMSE": base_rmse, "R2": base_r2
})


# --- Task 4: Tree-Depth Diagnostic ---
print("Running Tree-Depth Diagnostic...")
depths_to_test = [1, 2, 3, 4, 6, 8, 12, None]
tree_results = []

for d in depths_to_test:
    # 1. Initialize the Decision Tree
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    
    # 2. Train (fit) the tree using the training data
    # (Decision trees don't require scaled data, so we can use X_train)
    tree.fit(X_train, y_train)
    
    # 3. Predict on the Training set to see how well it memorized
    train_preds = tree.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    
    # 4. Predict on the Validation set to see how well it actually learned
    val_preds = tree.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    # 5. Record the results
    d_label = "unlimited" if d is None else d
    tree_results.append({
        "max_depth": d_label,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

# Convert to a DataFrame, round to 4 decimals, and save
tree_depth_df = pd.DataFrame(tree_results)
tree_depth_df = tree_depth_df.round(4)
tree_depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)

# Find the best depth (lowest Validation RMSE). 
# idxmin() naturally picks the first occurrence of the minimum value,
# meaning it breaks ties by choosing the smaller depth automatically!
best_depth_idx = tree_depth_df["val_rmse"].idxmin()
best_depth_val = tree_depth_df.loc[best_depth_idx, "max_depth"]
best_depth = None if best_depth_val == "unlimited" else int(best_depth_val)

print(f"Tree Diagnostic Complete! The optimal depth found is: {best_depth_val}")

# --- Task 5: Four Candidate Models ---
print("Training the four candidate models...")

# 1. Define the four models
# We use a Pipeline for Linear Regression to scale the data right before it hits the model
models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()), 
        ("model", LinearRegression())
    ]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# 2. Train each model, predict on validation, and score
for model_name, model_obj in models.items():
    # Fit on training data
    model_obj.fit(X_train, y_train)
    
    # Predict on validation data
    val_preds = model_obj.predict(X_val)
    
    # Calculate metrics
    mae = mean_absolute_error(y_val, val_preds)
    mse = mean_squared_error(y_val, val_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, val_preds)
    
    # Append to our tracking list from Task 3
    metrics_list.append({
        "model": model_name,
        "split": "val",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

# Convert to DataFrame, round, and save (we will overwrite this in Task 7 to add the test row)
metrics_df = pd.DataFrame(metrics_list)
metrics_df = metrics_df.round(4)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)


# --- Task 6: Learning-Rate Experiment ---
print("Running Learning-Rate Experiment for Gradient Boosting...")
lr_results = []
learning_rates_to_test = [0.01, 0.1, 1.0]

for lr in learning_rates_to_test:
    # Initialize Gradient Boosting with the current learning rate
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    
    # Train and predict
    gb.fit(X_train, y_train)
    val_preds = gb.predict(X_val)
    
    # Calculate RMSE and R2
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)
    
    lr_results.append({
        "learning_rate": lr,
        "val_rmse": rmse,
        "val_r2": r2
    })

# Convert to DataFrame, round, and save
lr_df = pd.DataFrame(lr_results)
lr_df = lr_df.round(4)
lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

print("Tasks 5 & 6 complete! outputs/part_b_metrics.csv and outputs/part_b_gb_learning_rate.csv generated.")

# --- Task 7: Cross-validated comparison and final model ---
print("Running Cross-Validation and Final Test Evaluation...")

# 1. Find the top 2 models by Validation RMSE (excluding the baseline)
# We filter out the baseline, sort by RMSE (lowest is best), and grab the top 2 names
model_only_metrics = metrics_df[metrics_df["model"] != "mean_baseline"]
top_2_names = model_only_metrics.sort_values("RMSE")["model"].head(2).tolist()
print(f"Top 2 models advancing to CV: {top_2_names}")

# 2. Prepare the combined Training + Validation data for Cross-Validation (840 + 180 = 1020 rows)
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# 3. Create the K-Fold object (5 splits, shuffled, random state 42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Run Cross-Validation on the top 2 models
cv_results = []
for name in top_2_names:
    model_obj = models[name] # Grab the model definition from our dictionary in Task 5
    
    # cross_val_score handles the splitting and training automatically using our KFold!
    # Default scoring for regressors in scikit-learn is R2.
    scores = cross_val_score(model_obj, X_train_val, y_train_val, cv=kf)
    
    cv_results.append({
        "model": name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    })

# Save CV results
cv_df = pd.DataFrame(cv_results).round(4)
cv_df.to_csv("outputs/part_b_cv.csv", index=False)

# 5. The Final Evaluation on the Test Set
# The absolute best model is the #1 model in our top_2_names list
best_model_name = top_2_names[0]
best_model = models[best_model_name] # This is already fitted on the 840 training rows from Task 5

# Predict on the locked-away Test Set
test_preds = best_model.predict(X_test)

# Calculate final test metrics
test_mae = mean_absolute_error(y_test, test_preds)
test_mse = mean_squared_error(y_test, test_preds)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_preds)

# Append this final test row to our metrics DataFrame and overwrite the CSV
test_row = {
    "model": best_model_name,
    "split": "test",
    "MAE": test_mae,
    "MSE": test_mse,
    "RMSE": test_rmse,
    "R2": test_r2
}
metrics_df = pd.concat([metrics_df, pd.DataFrame([test_row])], ignore_index=True)
metrics_df = metrics_df.round(4)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)

print(f"Part B Complete! The final winning model is: {best_model_name} with a Test R2 of {test_r2:.4f}")