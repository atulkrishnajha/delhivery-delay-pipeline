import pandas as pd
import numpy as np
from src.shipment_prep import clean_shipments, build_features
from src.shipment_prep import FEATURES

print("Starting Data Preparation...")

# 1. Read the raw data
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# 2. Clean the data (this also captures the medians we calculated)
clean, medians = clean_shipments(raw)

# 3. Engineer features
df = build_features(clean, hubs)

# 4. Create the target variable: is_delayed
# A parcel is delayed if its actual delivery hours strictly exceed promised hours.
df["is_delayed"] = (df["delivery_hours"] > df["promised_hours"]).astype(int)

# 5. Save the final modeling table
df.to_csv("data/modelling_table.csv", index=False)
print(f"Success! modelling_table.csv saved with {len(df)} rows.")

print("Generating EDA reports...")

# --- Step 5: System Health Metrics ---
raw_dedup = raw.drop_duplicates()

metrics = {}
metrics["raw_rows"] = len(raw)
metrics["duplicate_rows_removed"] = len(raw) - len(raw_dedup)
metrics["rows_after_dedup"] = len(raw_dedup)
metrics["missing_distance_km_before"] = raw_dedup["distance_km"].isna().sum()
metrics["missing_parcel_weight_before"] = raw_dedup["parcel_weight_kg"].isna().sum()
metrics["sentinel_weight_rows"] = (raw_dedup["parcel_weight_kg"] > 100).sum()
metrics["missing_declared_value_before"] = raw_dedup["declared_value"].isna().sum()
metrics["missing_payment_mode_before"] = raw_dedup["payment_mode"].isna().sum()
metrics["service_type_spellings_before"] = raw_dedup["service_type"].nunique()
metrics["service_type_spellings_after"] = df["service_type"].nunique()

metrics["declared_value_mean_inr"] = df["declared_value_inr"].mean()
metrics["declared_value_median_inr"] = df["declared_value_inr"].median()
metrics["log_declared_value_mean"] = df["log_declared_value"].mean()
metrics["log_declared_value_median"] = df["log_declared_value"].median()

metrics["median_distance_km"] = medians["distance_km"]
metrics["median_parcel_weight_kg"] = medians["parcel_weight_kg"]
metrics["median_declared_value_inr"] = medians["declared_value_inr"]

metrics["rows_after_hub_merge"] = len(df)

# Check for any remaining nulls in our ML features
cols_to_check = FEATURES + ["delivery_hours", "is_delayed"]
metrics["nulls_in_features_and_target"] = df[cols_to_check].isna().sum().sum()

metrics["delayed_count"] = df["is_delayed"].sum()
metrics["delay_rate"] = df["is_delayed"].mean()

# Calculate outliers using the 1.5 * IQR rule
q1 = df["delivery_hours"].quantile(0.25)
q3 = df["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

metrics["hours_q1"] = q1
metrics["hours_q3"] = q3
metrics["hours_lower_bound"] = lower_bound
metrics["hours_upper_bound"] = upper_bound
metrics["hours_outlier_count"] = ((df["delivery_hours"] < lower_bound) | (df["delivery_hours"] > upper_bound)).sum()

# Save metrics to CSV
checks_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
checks_df["value"] = checks_df["value"].apply(lambda x: round(x, 4) if isinstance(x, float) else x)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)


# --- Step 6: Business Evidence (Delay by Group) ---
# Create distance buckets
df["distance_band"] = pd.cut(
    df["distance_km"], 
    bins=[0, 200, 400, 600, 2000], 
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True
)

group_results = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    grouped = df.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()
    
    grouped.rename(columns={dim: "group"}, inplace=True)
    grouped.insert(0, "dimension", dim)
    group_results.append(grouped)
    
delay_by_group = pd.concat(group_results, ignore_index=True)
delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)
delay_by_group["mean_hours"] = delay_by_group["mean_hours"].round(4)
delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)


# --- Step 7: Correlation Matrix ---
# Select only numeric columns to calculate math correlations
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().round(4)

# Move the row names (index) into the first column as requested by the brief
corr_matrix.reset_index(inplace=True) 
corr_matrix.to_csv("outputs/part_a_correlations.csv", index=False)

print("Part A complete! Outputs saved in outputs/ folder.")