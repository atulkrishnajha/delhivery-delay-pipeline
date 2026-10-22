import pandas as pd
import numpy as np

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
def clean_shipments(raw, medians=None):
    """Cleans the raw shipment data step-by-step."""
    # Step 1: Drop exact duplicate rows
    clean_df = raw.drop_duplicates().copy()
    
    # Step 2: service_type - strip spaces and title case
    clean_df["service_type"] = clean_df["service_type"].str.strip().str.title()
    
    # Step 3: payment_mode - fill missing with "Unknown"
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")
    
    # Step 4: declared_value - remove ₹ and commas, convert to float
    # We will use string replacement and astype(float)
    clean_df["declared_value_inr"] = (
        clean_df["declared_value"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    clean_df = clean_df.drop(columns=["declared_value"])
    
    # Step 5: dispatch_time - convert to datetime
    clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"])
    
    # Step 6: parcel_weight_kg - over 100 is a sentinel (scanner fault), treat as missing (NaN)
    clean_df.loc[clean_df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    # Step 7: Calculate or use provided medians to fill missing values
    if medians is None:
        medians = {
            "distance_km": clean_df["distance_km"].median(),
            "parcel_weight_kg": clean_df["parcel_weight_kg"].median(),
            "declared_value_inr": clean_df["declared_value_inr"].median()
        }
        
    clean_df["distance_km"] = clean_df["distance_km"].fillna(medians["distance_km"])
    clean_df["parcel_weight_kg"] = clean_df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(medians["declared_value_inr"])
    
    return clean_df, medians
def build_features(clean, hubs):
    """Merges hub data and engineers new numeric features for ML."""
    # 1. Left merge to attach hub details based on 'hub_id'
    # A 'left' merge ensures we don't lose any shipments even if a hub was missing.
    df = clean.merge(hubs, on="hub_id", how="left")
    
    # 2. promised_hours: Use our dictionary to map 'Express' -> 36, etc.
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    
    # 3. is_cod: True becomes 1, False becomes 0
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    
    # 4. is_weekend: In pandas, Monday=0, Sunday=6. Weekend is >= 5.
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    
    # 5. dispatch_hour: Extract just the hour (0-23) from the timestamp
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    
    # 6. log_declared_value: Natural logarithm of the money value
    # We do this because monetary values can be highly skewed (a few very expensive items).
    # Log transforms pull these extreme values closer, helping regression models learn better.
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    
    return df