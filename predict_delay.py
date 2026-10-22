import pandas as pd
import joblib
from src.shipment_prep import clean_shipments, build_features

print("Running Prediction Pipeline on New Shipments...")

# 1. Load all required files
new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
hub_segments = pd.read_csv("outputs/hub_segments.csv")

# Load model metadata
threshold = pd.read_csv("models/delay_threshold.csv").iloc[0]["threshold"]
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute_df = pd.read_csv("models/impute_values.csv")
# Convert impute_df back into a dictionary
saved_medians = dict(zip(impute_df["column"], impute_df["median"]))

# Load the brain
model = joblib.load("models/delay_model.pkl")

# 2. Process the new data using our exact Part A logic, but passing the saved medians!
# We keep the 'shipment_id' safe so we can use it in the final report.
clean_new, _ = clean_shipments(new_raw, medians=saved_medians)
features_new = build_features(clean_new, hubs)

# Force the exact column order the model expects
X_new = features_new[feature_order]

# 3. Predict the delay probability (rounded to 4 decimals)
probs = model.predict_proba(X_new)[:, 1]
probs_rounded = [round(p, 4) for p in probs]

# 4. Build the final output dataframe
results = pd.DataFrame({
    "shipment_id": clean_new["shipment_id"],
    "hub_id": clean_new["hub_id"],
    "delay_probability": probs_rounded
})

# Add the hub segments
results = results.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")

# Set the flagged column based on our threshold
results["flagged"] = (results["delay_probability"] >= threshold).astype(int)

# 5. Apply the Business Action Logic
def determine_action(row):
    if row["flagged"] == 1:
        if row["segment"] == "Escalate":
            return "Escalate to hub manager"
        else:
            return "Proactive customer call"
    else:
        return "No action"

results["action"] = results.apply(determine_action, axis=1)

# Sort columns strictly as requested
final_output = results[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
final_output.to_csv("outputs/new_shipment_actions.csv", index=False)

print(f"Success! {len(final_output)} new parcels processed. Actions saved to outputs/new_shipment_actions.csv.")