import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.shipment_prep import NOISE_COLUMNS

print("Starting Part C: Classification Modeling...")

# Load the clean modeling table
df = pd.read_csv("data/modelling_table.csv")

# Inputs (X) and target (y)
X = df[FEATURES]
y = df["is_delayed"]

# --- Task 1: Stratified Split ---
# 1. Split off 70% for training. Note the new stratify=y parameter!
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Split the remaining 30% in half for validation and test.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
print(f"Delayed parcels: Train={y_train.sum()}, Val={y_val.sum()}, Test={y_test.sum()}")


# --- Task 2: Baseline and Three Classifiers ---
print("Training initial classifiers...")
results = []

# Helper function to calculate metrics and append to our results list
def record_metrics(model_name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # The brief states to manually write precision as 0 for the blind baseline to avoid divide-by-zero errors
    if model_name == "blind_all_on_time":
        precision = 0.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        
    results.append({
        "model": model_name,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    })

# 1. Blind Baseline (Predicts 0 for everything)
y_pred_blind = np.zeros_like(y_val)
record_metrics("blind_all_on_time", y_val, y_pred_blind)

# 2. Logistic Regression (Requires scaling via Pipeline)
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])
log_reg.fit(X_train, y_train)
record_metrics("logistic_regression", y_val, log_reg.predict(X_val))

# 3. Decision Tree Diagnostic (Find the best depth by F1-score)
best_dt_f1 = -1
best_dt_depth = None
best_dt_preds = None

for d in [2, 3, 4, 5, 6, 8, 10]:
    dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_val)
    f1 = f1_score(y_val, preds)
    
    # If this depth's F1 is better than our current best, save it!
    # By strictly using > (not >=), ties automatically favor the smaller depth.
    if f1 > best_dt_f1:
        best_dt_f1 = f1
        best_dt_depth = d
        best_dt_preds = preds

# Record the best decision tree
record_metrics(f"decision_tree_depth_{best_dt_depth}", y_val, best_dt_preds)

# 4. Random Forest Default
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
record_metrics("random_forest_default", y_val, rf.predict(X_val))

# Save results to CSV
models_df = pd.DataFrame(results)
# Round metrics to 4 decimal places as per brief requirements
metrics_cols = ["accuracy", "precision", "recall", "f1"]
models_df[metrics_cols] = models_df[metrics_cols].round(4)
models_df.to_csv("outputs/part_c_models.csv", index=False)

print("Tasks 1 & 2 complete! outputs/part_c_models.csv generated.")

# --- Task 3: Which columns carry signal? ---
print("Evaluating Feature Importance and Signal...")

# 1. Fit a default forest on all 17 columns to get its internal importances
rf_feat = RandomForestClassifier(n_estimators=100, random_state=42)
rf_feat.fit(X_train, y_train)
importances = rf_feat.feature_importances_

# 2. Fit SelectKBest (F-test) on all columns
selector_all = SelectKBest(f_classif, k="all")
selector_all.fit(X_train, y_train)
f_scores = selector_all.scores_

# 3. Create the feature scores table
feature_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": importances,
    "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
})

# Calculate f_rank (1 is the highest score)
feature_df["f_rank"] = feature_df["f_score"].rank(ascending=False, method="min").astype(int)
feature_df = feature_df.sort_values("f_rank")

# Round and save
feature_df["f_score"] = feature_df["f_score"].round(4)
feature_df["rf_importance"] = feature_df["rf_importance"].round(4)
feature_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

# 4. Test different 'k' feature counts to see how the model reacts
k_results = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    
    k_results.append({
        "k": k, 
        "val_f1": f1_score(y_val, preds, zero_division=0)
    })

pd.DataFrame(k_results).round(4).to_csv("outputs/part_c_selectk.csv", index=False)


# --- Task 4: Tuning the Random Forest ---
print("Tuning the Forest on the Top 7 Features...")

# 1. Transform the data down to just the top 7 features based on the Training set
selector_7 = SelectKBest(f_classif, k=7)
X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val) # Remember, no fitting on validation data!

# 2. Define the grid of hyperparameters to test
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# 3. Run GridSearchCV (Tests all 3x3x3 = 27 combinations, 5 folds each = 135 fits)
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train_7, y_train)

# 4. Run RandomizedSearchCV (Tests just 9 random combinations, 5 folds each = 45 fits)
rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid, n_iter=9, cv=5, random_state=42
)
rand_search.fit(X_train_7, y_train)

# 5. The Default Forest on the exact same 7 features (for a fair baseline comparison)
rf_default_7 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default_7.fit(X_train_7, y_train)

# 6. Compile tuning results
tuning_results = [
    {
        "search": "grid_search",
        "candidates": 27, "cv_folds": 5, "total_fits": 135,
        "best_params": str(grid.best_params_),
        "best_cv_accuracy": round(grid.best_score_, 4),
        "val_f1": round(f1_score(y_val, grid.predict(X_val_7)), 4)
    },
    {
        "search": "randomized_search",
        "candidates": 9, "cv_folds": 5, "total_fits": 45,
        "best_params": str(rand_search.best_params_),
        "best_cv_accuracy": round(rand_search.best_score_, 4),
        "val_f1": round(f1_score(y_val, rand_search.predict(X_val_7)), 4)
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1, "cv_folds": 0, "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": "", # Left blank intentionally
        "val_f1": round(f1_score(y_val, rf_default_7.predict(X_val_7)), 4)
    }
]

pd.DataFrame(tuning_results).to_csv("outputs/part_c_tuning.csv", index=False)
print("Tasks 3 & 4 complete! Feature scores and tuning tables generated.")

# --- Task 5: Choosing the final configuration ---
print("Selecting the Final Classification Model...")

# 1. We evaluate the tuning_results list we built in Task 4.
# We want the highest val_f1. In a tie, the brief prefers: default > grid > randomized.
tie_breaker_scores = {"default_forest_same_7_features": 3, "grid_search": 2, "randomized_search": 1}

# Sort by val_f1 (descending) and then by our tie-breaker score (descending)
tuning_results_sorted = sorted(
    tuning_results, 
    key=lambda x: (x["val_f1"], tie_breaker_scores[x["search"]]), 
    reverse=True
)

best_config = tuning_results_sorted[0]
best_search_name = best_config["search"]
print(f"Winning configuration: {best_search_name} with Validation F1: {best_config['val_f1']}")

# 2. Retrieve the exact parameters the winning model used
if best_search_name == "grid_search":
    final_rf = RandomForestClassifier(random_state=42, **grid.best_params_)
    final_params = grid.best_params_
elif best_search_name == "randomized_search":
    final_rf = RandomForestClassifier(random_state=42, **rand_search.best_params_)
    final_params = rand_search.best_params_
else:
    final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    final_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

# 3. Build and fit the final Pipeline on all 17 columns of the Training set
final_pipeline = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", final_rf)
])
final_pipeline.fit(X_train, y_train)


# --- Task 6: Alert Threshold ---
print("Calculating Alert Thresholds...")

# 1. Get probabilities for the Validation set 
# predict_proba returns [Prob_Not_Delayed, Prob_Delayed]. We want column index 1.
val_probs = final_pipeline.predict_proba(X_val)[:, 1]

thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thresh_results = []
chosen_threshold = 0.1 # Default fallback if nothing hits 0.80 recall

for t in thresholds_to_test:
    # Predict 1 if the probability is >= the threshold, else 0
    t_preds = (val_probs >= t).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_val, t_preds).ravel()
    
    thresh_results.append({
        "threshold": t, 
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": accuracy_score(y_val, t_preds), 
        "precision": precision_score(y_val, t_preds, zero_division=0), 
        "recall": recall_score(y_val, t_preds), 
        "f1": f1_score(y_val, t_preds, zero_division=0)
    })

thresh_df = pd.DataFrame(thresh_results).round(4)
thresh_df.to_csv("outputs/part_c_thresholds.csv", index=False)

# 2. Operations rule: highest threshold with validation recall >= 0.80
valid_thresholds = thresh_df[thresh_df["recall"] >= 0.80]
if not valid_thresholds.empty:
    chosen_threshold = valid_thresholds["threshold"].max()
    
print(f"Operations Rule applied! Chosen Alert Threshold: {chosen_threshold}")


# --- Task 7: One Test Evaluation ---
print("Running Final Test Evaluation...")

# 1. Predict probabilities for the locked-away Test set
test_probs = final_pipeline.predict_proba(X_test)[:, 1]

# 2. Apply our chosen threshold to these test probabilities
test_preds = (test_probs >= chosen_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

test_results = pd.DataFrame([{
    "threshold": chosen_threshold,
    "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    "accuracy": round(accuracy_score(y_test, test_preds), 4),
    "precision": round(precision_score(y_test, test_preds, zero_division=0), 4),
    "recall": round(recall_score(y_test, test_preds), 4),
    "f1": round(f1_score(y_test, test_preds, zero_division=0), 4)
}])
test_results.to_csv("outputs/part_c_final_test.csv", index=False)

# 3. Save the final configuration for packaging later
max_d = final_params.get("max_depth")
config_df = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": final_params.get("n_estimators"),
    "max_depth": "unlimited" if max_d is None else max_d,
    "min_samples_leaf": final_params.get("min_samples_leaf"),
    "threshold": chosen_threshold,
    "chosen_from": best_search_name
}])
config_df.to_csv("outputs/part_c_final_config.csv", index=False)

print("Part C Complete! Final model configuration and test evaluations saved.")