"""
Fit a clean baseline curve R_clean(HP) for each asset in air_filter_data.csv using Isotonic Regression with linear extrapolation.

- Loads HydraulicHorsepower and AirFilterRestriction columns efficiently (chunked).
- Bins HydraulicHorsepower, computes 5th percentile AirFilterRestriction per bin (lower envelope).
- Fits Isotonic Regression to (HP_bin_center, 5th percentile Restriction) pairs.
- Ensures monotonicity: restriction increases with HP.
- Forces extrapolation to reach max_restriction at max_hp to cover full operating range.
- For each asset present in both air_filter_data.csv and asset_limits.csv:
    - Saves model with extrapolation parameters as /api/{asset}/model.pkl
    - Outputs /api/{asset}/{asset}_model_bins.csv with (hp_bin_center, 5th percentile restriction)
    - Plots baseline as /api/{asset}/{asset}_model_baseline.png for visual inspection.
"""

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import joblib
import matplotlib.pyplot as plt
import os
import csv

DATA_PATH = "data/air_filter_data.csv"
LIMITS_PATH = "data/asset_limits.csv"
CHUNKSIZE = 100_000
HP_BIN_WIDTH = 50  # Bin width for HP (adjust based on your data density)
PERCENTILE = 5  # Lower envelope

# Step 1: Load asset limits
limits_df = pd.read_csv(LIMITS_PATH)
def normalize_asset_id(val):
    # Handles both float and string asset IDs, strips .0 if present
    try:
        return str(int(float(val)))
    except Exception:
        return str(val)

asset_limits = {
    normalize_asset_id(row['asset']): {
        'max_restriction': row['Max_AirFilterRestriction'],
        'max_hp': row['Max_Horsepower']
    }
    for _, row in limits_df.iterrows()
}

# Step 2: Identify all assets in both air_filter_data.csv and asset_limits.csv
print("Scanning available assets...")
asset_ids = set()
all_data_asset_ids = set()
for chunk in pd.read_csv(DATA_PATH, usecols=["asset"], chunksize=CHUNKSIZE):
    chunk_ids = set(normalize_asset_id(a) for a in chunk["asset"].unique())
    all_data_asset_ids.update(chunk_ids)
    asset_ids.update(chunk_ids)
print(f"Unique asset IDs in air_filter_data.csv: {sorted(all_data_asset_ids)}")
print(f"Unique asset IDs in asset_limits.csv: {sorted(asset_limits.keys())}")
asset_ids = sorted([aid for aid in asset_ids if aid in asset_limits])
print(f"Intersection asset IDs to process: {asset_ids}")

# Step 3: Process each asset
for asset_id in asset_ids:
    print(f"\n{'='*60}\nProcessing asset {asset_id}\n{'='*60}")
    # Prepare output directory
    asset_dir = os.path.join("api", asset_id)
    os.makedirs(asset_dir, exist_ok=True)

    # Load data for this asset
    hp_list = []
    restriction_list = []
    for chunk in pd.read_csv(DATA_PATH, usecols=["asset", "HydraulicHorsepower", "AirFilterRestriction"], chunksize=CHUNKSIZE):
        mask = chunk["asset"].astype(str) == asset_id
        if mask.any():
            hp_list.append(chunk.loc[mask, "HydraulicHorsepower"].values)
            restriction_list.append(chunk.loc[mask, "AirFilterRestriction"].values)
    if not hp_list:
        print(f"No data found for asset {asset_id}, skipping.")
        continue
    hp = np.concatenate(hp_list)
    restriction = np.concatenate(restriction_list)

    max_hp_asset = asset_limits[asset_id]['max_hp']
    max_restriction_asset = asset_limits[asset_id]['max_restriction']

    print(f"Loaded {len(hp)} data points for asset {asset_id}")
    print(f"HP range: {np.min(hp):.2f} to {np.max(hp):.2f}")
    print(f"Restriction range: {np.min(restriction):.2f} to {np.max(restriction):.2f}")
    print(f"Asset HP limit: {max_hp_asset} (from asset_limits.csv)")
    print(f"Asset Restriction limit: {max_restriction_asset} (from asset_limits.csv)")

    # Step 4: Bin HydraulicHorsepower and compute 5th percentile Restriction per bin
    hp_min, hp_max = np.min(hp), np.max(hp)
    bins = np.arange(hp_min, max(hp_max, max_hp_asset) + HP_BIN_WIDTH, HP_BIN_WIDTH)
    bin_indices = np.digitize(hp, bins) - 1  # bins are right-exclusive

    bin_centers = []
    bin_restriction = []
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        bin_center = (bins[i] + bins[i+1]) / 2
        if np.any(mask):
            perc = np.percentile(restriction[mask], PERCENTILE)
            bin_centers.append(bin_center)
            bin_restriction.append(perc)
        else:
            bin_centers.append(bin_center)
            bin_restriction.append(np.nan)

    bin_centers = np.array(bin_centers)
    bin_restriction = np.array(bin_restriction)

    # Remove bins with nan before fitting models
    fit_mask = ~np.isnan(bin_restriction)
    fit_bin_centers = bin_centers[fit_mask]
    fit_bin_restriction = bin_restriction[fit_mask]

    print(f"Created {len(fit_bin_centers)} bins with data for model fitting")
    print(f"HP bin range (fitted): {fit_bin_centers.min():.2f} to {fit_bin_centers.max():.2f}")
    print(f"Restriction range (5th percentile): {fit_bin_restriction.min():.2f} to {fit_bin_restriction.max():.2f}")

    # Output CSV of (hp_bin_center, 5th percentile restriction)
    bins_csv_path = os.path.join(asset_dir, f"{asset_id}_model_bins.csv")
    with open(bins_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hp_bin_center", "restriction_5th_percentile"])
        for center, val in zip(bin_centers, bin_restriction):
            writer.writerow([center, val])

    # Step 5: Fit Isotonic Regression (HP → Restriction, increasing)
    print("\nFitting Isotonic Regression (HP → Restriction)...")
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(fit_bin_centers, fit_bin_restriction)

    # Get predictions for fitted range
    iso_pred = iso.predict(fit_bin_centers)

    print(f"Isotonic model fitted:")
    print(f"  Input range (HP): {fit_bin_centers.min():.2f} to {fit_bin_centers.max():.2f}")
    print(f"  Output range (Restriction): {iso_pred.min():.2f} to {iso_pred.max():.2f}")

    # Step 6: Determine initial linear extrapolation parameters
    print("\nComputing initial linear extrapolation...")

    slopes = np.diff(iso_pred) / np.diff(fit_bin_centers)
    SLOPE_THRESHOLD = 0.001
    still_rising = slopes > SLOPE_THRESHOLD

    if np.any(still_rising):
        N_POINTS_FOR_EXTRAP = 10
        last_rising_indices = np.where(still_rising)[0]
        if len(last_rising_indices) >= N_POINTS_FOR_EXTRAP:
            extrap_indices = last_rising_indices[-N_POINTS_FOR_EXTRAP:]
        else:
            extrap_indices = last_rising_indices
        X_extrap = fit_bin_centers[extrap_indices]
        y_extrap = iso_pred[extrap_indices]
        extrapolation_slope, extrapolation_intercept = np.polyfit(X_extrap, y_extrap, 1)
        print(f"Initial linear extrapolation from last {len(extrap_indices)} rising points")
        print(f"  HP range used: {X_extrap.min():.2f} to {X_extrap.max():.2f}")
        print(f"  Initial slope: {extrapolation_slope:.6f} Restriction per HP")
        print(f"  Initial intercept: {extrapolation_intercept:.6f}")
    else:
        print("Warning: No significant rising trend detected, using last two points for initial extrapolation")
        X_extrap = fit_bin_centers[-2:]
        y_extrap = iso_pred[-2:]
        extrapolation_slope, extrapolation_intercept = np.polyfit(X_extrap, y_extrap, 1)
        print(f"  Initial slope: {extrapolation_slope:.6f} Restriction per HP")
        print(f"  Initial intercept: {extrapolation_intercept:.6f}")

    # Step 7: Force extrapolation to cover full range
    print("\nAdjusting extrapolation to cover full operating range...")

    hp_at_extrap_start = fit_bin_centers.max()
    r_at_extrap_start = iso.predict([[hp_at_extrap_start]])[0]
    restriction_at_max_hp_natural = extrapolation_slope * max_hp_asset + extrapolation_intercept

    print(f"Natural extrapolation would reach R={restriction_at_max_hp_natural:.2f} at HP={max_hp_asset}")

    if restriction_at_max_hp_natural < max_restriction_asset:
        print(f"\n[FORCING STEEPER SLOPE] Natural extrapolation insufficient")
        print(f"  Need to reach: R={max_restriction_asset} at HP={max_hp_asset}")
        extrapolation_slope = (max_restriction_asset - r_at_extrap_start) / (max_hp_asset - hp_at_extrap_start)
        extrapolation_intercept = r_at_extrap_start - extrapolation_slope * hp_at_extrap_start
        print(f"  Adjusted slope: {extrapolation_slope:.6f} Restriction per HP")
        print(f"  Adjusted intercept: {extrapolation_intercept:.6f}")
        print(f"  Extrapolation now reaches R={max_restriction_asset:.2f} at HP={max_hp_asset}")
        hp_at_max_restriction = max_hp_asset
    else:
        print(f"Natural extrapolation is sufficient")
        if extrapolation_slope > 0:
            hp_at_max_restriction = (max_restriction_asset - extrapolation_intercept) / extrapolation_slope
            print(f"  Extrapolation reaches max restriction at HP={hp_at_max_restriction:.2f}")
        else:
            hp_at_max_restriction = np.inf

    # Step 8: Create prediction function
    def predict_restriction_from_hp(hp_values, iso_model, max_fitted_hp, slope, intercept, max_restriction_limit):
        hp_values = np.atleast_1d(hp_values)
        predictions = np.zeros_like(hp_values, dtype=float)
        for i, hp_val in enumerate(hp_values):
            if hp_val <= max_fitted_hp:
                predictions[i] = iso_model.predict([[hp_val]])[0]
            else:
                predictions[i] = slope * hp_val + intercept
            predictions[i] = min(predictions[i], max_restriction_limit)
        return predictions

    # Step 9: Save model with extrapolation parameters
    model_data = {
        'iso': iso,
        'max_hp_fitted': fit_bin_centers.max(),
        'min_hp_fitted': fit_bin_centers.min(),
        'extrap_slope': extrapolation_slope,
        'extrap_intercept': extrapolation_intercept,
        'max_restriction_fitted': iso_pred.max(),
        'min_restriction_fitted': iso_pred.min(),
        'max_hp_asset': max_hp_asset,
        'max_restriction_asset': max_restriction_asset,
        'hp_at_max_restriction': hp_at_max_restriction
    }
    model_path = os.path.join(asset_dir, "model.pkl")
    joblib.dump(model_data, model_path)

    print(f"\n{'='*60}")
    print(f"Model saved to {model_path}")
    print(f"{'='*60}")
    print(f"Fitted HP range: {model_data['min_hp_fitted']:.2f} to {model_data['max_hp_fitted']:.2f}")
    print(f"Fitted Restriction range: {model_data['min_restriction_fitted']:.2f} to {model_data['max_restriction_fitted']:.2f}")
    print(f"Asset limits: HP={max_hp_asset}, Restriction={max_restriction_asset} (from asset_limits.csv)")
    print(f"Model coverage: HP up to {max_hp_asset}, Restriction up to {max_restriction_asset}")

    # Step 10: Plot for visual inspection
    print("\nCreating visualization...")
    plt.figure(figsize=(14, 8))
    plt.scatter(hp, restriction, s=1, alpha=0.1, label="All Data", color='lightblue')
    plt.plot(fit_bin_centers, fit_bin_restriction, 'go', markersize=6, label="5th Percentile per Bin", zorder=5)
    hp_plot = np.linspace(fit_bin_centers.min(), fit_bin_centers.max(), 500)
    restriction_plot = iso.predict(hp_plot)
    plt.plot(hp_plot, restriction_plot, 'r-', linewidth=2, label="Isotonic Baseline (fitted)", zorder=4)
    hp_extrap = np.linspace(fit_bin_centers.max(), max_hp_asset, 100)
    restriction_extrap = extrapolation_slope * hp_extrap + extrapolation_intercept
    restriction_extrap = np.minimum(restriction_extrap, max_restriction_asset)
    plt.plot(hp_extrap, restriction_extrap, 'r--', linewidth=2, label="Linear Extrapolation (forced)", zorder=4)
    plt.axvline(fit_bin_centers.max(), color='orange', linestyle=':', linewidth=1.5,
                label=f"Extrapolation starts at HP={fit_bin_centers.max():.0f}", zorder=3)
    plt.axhline(max_restriction_asset, color='purple', linestyle='-.', linewidth=1.5, alpha=0.5,
                label=f"Asset Restriction Limit ({max_restriction_asset})", zorder=3)
    plt.axvline(max_hp_asset, color='brown', linestyle='-.', linewidth=1.5, alpha=0.5,
                label=f"Asset HP Limit ({max_hp_asset})", zorder=3)
    if hp_at_max_restriction <= max_hp_asset:
        plt.plot(hp_at_max_restriction, max_restriction_asset, 'rs', markersize=10,
                 label=f"Reaches Max R at HP={hp_at_max_restriction:.0f}", zorder=6)
    plt.xlabel("HydraulicHorsepower", fontsize=12)
    plt.ylabel("AirFilterRestriction", fontsize=12)
    plt.title(f"Clean Baseline Curve Fit: R_clean(HP) - Asset {asset_id}", fontsize=14)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_hp_asset * 1.05)
    plt.ylim(0, max_restriction_asset * 1.1)
    plt.tight_layout()
    plot_path = os.path.join(asset_dir, f"{asset_id}_model_baseline.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Baseline plot saved to {plot_path}")
    print(f"Bin summary saved to {bins_csv_path}")

    # Step 11: Test the prediction function
    print("\n" + "="*60)
    print("Testing R_clean(HP) prediction function:")
    print("="*60)
    test_hps = [200, 500, 1000, 1200, 1500, 1800, 2000, int(max_hp_asset)]
    for hp_test in test_hps:
        r_pred = predict_restriction_from_hp(hp_test, iso, model_data['max_hp_fitted'],
                                             extrapolation_slope, extrapolation_intercept,
                                             max_restriction_asset)
        in_range = "✓ (fitted)" if hp_test <= model_data['max_hp_fitted'] else "✗ (extrapolated)"
        print(f"HP={hp_test:4.0f} → R_clean={r_pred[0]:6.2f}  {in_range}")

    print("\n" + "="*60)
    print(f"Model fitting complete for asset {asset_id}!")
    print("="*60)
    print("\nNote: Extrapolation beyond fitted data assumes linear relationship")
    print("between HP and Restriction up to system limits. This is an assumption")
    print("since no clean baseline data exists in the high restriction range.")

print("\nAll assets processed.")

# Combine all asset model.pkl files into a single all_models.pkl
import joblib
import os

def combine_all_models(asset_ids, base_dir="api"):
    models = {}
    for asset_id in asset_ids:
        model_path = os.path.join(base_dir, asset_id, "model.pkl")
        if os.path.exists(model_path):
            models[asset_id] = joblib.load(model_path)
        else:
            print(f"Warning: {model_path} not found, skipping.")
    output_path = os.path.join(base_dir, "all_models.pkl")
    joblib.dump(models, output_path)
    print(f"Combined model file saved to {output_path}")

combine_all_models(asset_ids)