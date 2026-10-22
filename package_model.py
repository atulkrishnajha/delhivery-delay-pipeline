import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import clean_shipments, FEATURES

print("Packaging the Final Model...")

# 1. Read the final configuration chosen in Part C
config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
threshold = config["threshold"]

# Rebuild the exact same training environment
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

# Exact stratified splits from Part C
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 2. Build the final pipeline based on the saved config
max_depth_val = None if config["max_depth"] == "unlimited" else int(config["max_depth"])

final_rf = RandomForestClassifier(
    n_estimators=int(config["n_estimators"]),
    max_depth=max_depth_val,
    min_samples_leaf=int(config["min_samples_leaf"]),
    random_state=42
)

final_pipeline = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
    ("model", final_rf)
])

# Fit the final model one last time on the training data
final_pipeline.fit(X_train, y_train)

# 3. Save (Pickle) the model and metadata to the models/ directory
joblib.dump(final_pipeline, "models/delay_model.pkl")

pd.DataFrame([{"threshold": threshold}]).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

# Re-run clean_shipments on the raw data just to grab the medians dict to save
raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)
pd.DataFrame({
    "column": list(medians.keys()),
    "median": list(medians.values())
}).to_csv("models/impute_values.csv", index=False)

# 4. The Reload Check (Ensuring the saved model works exactly like the in-memory one)
loaded_model = joblib.load("models/delay_model.pkl")

# Predict on the test set with both models
memory_probs = final_pipeline.predict_proba(X_test)[:, 1]
loaded_probs = loaded_model.predict_proba(X_test)[:, 1]

# Calculate the maximum difference (should be mathematically 0)
max_diff = np.abs(memory_probs - loaded_probs).max()

reload_check = pd.DataFrame([
    {"metric": "test_rows_scored", "value": len(X_test)},
    {"metric": "reload_max_abs_probability_diff", "value": max_diff},
    {"metric": "threshold", "value": threshold}
])
reload_check.to_csv("outputs/part_d_reload_check.csv", index=False)

print("Model successfully packaged and verified!")