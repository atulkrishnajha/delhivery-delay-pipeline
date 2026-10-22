import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations

print("Starting Part D: Hub Segmentation...")

# Load the clean modeling table
df = pd.read_csv("data/modelling_table.csv")

# --- Task 1: Hub Profile ---
print("Building Hub Profiles...")

# Group by hub_id and calculate the specific metrics requested
hub_profile = df.groupby("hub_id").agg(
    shipments=("is_delayed", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).reset_index()

# Ensure it is sorted perfectly by ascending hub_id (HUB01 -> HUB24)
hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)

# Save to CSV
hub_profile_out = hub_profile.round(4)
hub_profile_out.to_csv("outputs/hub_profile.csv", index=False)


# --- Task 2: K-Means Segmentation ---
print("Running K-Means Clustering...")

# 1. Select the 4 profile columns to scale (excluding hub_id and shipments count)
cols_to_cluster = [
    "mean_delivery_hours", 
    "delay_rate", 
    "mean_distance_km", 
    "mean_n_stops"
]

# 2. Scale the data so all metrics carry equal weight
scaler = StandardScaler()
profile_scaled_array = scaler.fit_transform(hub_profile[cols_to_cluster])

# 3. Fit K-Means with exactly 3 clusters (as requested by the business)
# Using random_state=42 and n_init=10 exactly as the brief strictly requires.
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_3.fit(profile_scaled_array)

print("Tasks 1 & 2 complete! outputs/hub_profile.csv generated and K=3 fitted.")

# --- Task 3: Separation evidence ---
print("Evaluating Cluster Separation (K=2 to 5)...")

separation_results = []
n_hubs = len(hub_profile)

# We test K=2, 3, 4, 5
for k in [2, 3, 4, 5]:
    # Fit K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(profile_scaled_array)
    
    # Calculate cluster sizes and format them as descending (e.g., "9-8-7")
    sizes = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
    sizes_str = "-".join(map(str, sizes))
    
    # Calculate distances for all possible pairs of hubs (276 pairs total)
    within_distances = []
    across_distances = []
    
    for i, j in combinations(range(n_hubs), 2):
        # Euclidean distance between the two scaled hub profiles
        dist = np.linalg.norm(profile_scaled_array[i] - profile_scaled_array[j])
        
        if labels[i] == labels[j]:
            within_distances.append(dist)
        else:
            across_distances.append(dist)
            
    # Calculate means
    mean_within = np.mean(within_distances) if within_distances else 1.0 # prevent divide by zero
    mean_across = np.mean(across_distances)
    
    ratio = mean_across / mean_within
    
    separation_results.append({
        "K": k,
        "across_over_within": round(ratio, 4),
        "cluster_sizes": sizes_str
    })

pd.DataFrame(separation_results).to_csv("outputs/part_d_separation.csv", index=False)


# --- Task 4: Segment names and tier check ---
print("Assigning Business Segments and Finding Mismatches...")

# 1. We use our K=3 labels from Task 2
hub_profile["cluster_label"] = kmeans_3.labels_

# 2. Find the average delay rate for each cluster to rank them
cluster_delay_rates = hub_profile.groupby("cluster_label")["delay_rate"].mean().sort_values()

# The cluster with the lowest delay rate gets index 0, middle gets 1, highest gets 2 in this sorted list
ranked_cluster_ids = cluster_delay_rates.index.tolist()

# Map the mathematical labels (0, 1, 2) to the business names
segment_map = {
    ranked_cluster_ids[0]: "Standard",
    ranked_cluster_ids[1]: "Watch",
    ranked_cluster_ids[2]: "Escalate"
}
hub_profile["segment"] = hub_profile["cluster_label"].map(segment_map)

# 3. Bring in the original Hubs data to get City and Tier
hubs_raw = pd.read_csv("data/hubs.csv")
hub_segments = hub_profile.merge(hubs_raw[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")

# Save hub_segments.csv
out_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]]
out_segments.to_csv("outputs/hub_segments.csv", index=False)

# 4. Generate the Segment Profile (Averages for the newly named segments)
segment_profile = hub_segments.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()

# Sort segment profile logically: Standard, Watch, Escalate
segment_profile["segment_cat"] = pd.Categorical(
    segment_profile["segment"], categories=["Standard", "Watch", "Escalate"], ordered=True
)
segment_profile = segment_profile.sort_values("segment_cat").drop(columns="segment_cat")
segment_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

# 5. Find the Mismatches (Where actual performance doesn't match the Tier label)
tier_to_expected_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
hub_segments["expected_segment"] = hub_segments["hub_tier"].map(tier_to_expected_segment)

mismatches = hub_segments[hub_segments["segment"] != hub_segments["expected_segment"]]
out_mismatches = mismatches[["hub_id", "segment", "hub_city", "hub_tier"]]
out_mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print("Tasks 3 & 4 complete! Hub segments, profiles, and mismatches generated.")