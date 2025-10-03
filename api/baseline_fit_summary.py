import os
import pandas as pd
import matplotlib.pyplot as plt

def load_baseline_metrics(asset_dir="api"):
    summary = []
    for asset_id in os.listdir(asset_dir):
        asset_path = os.path.join(asset_dir, asset_id)
        bins_csv = os.path.join(asset_path, f"{asset_id}_model_bins.csv")
        plot_png = os.path.join(asset_path, f"{asset_id}_model_baseline.png")
        if os.path.isdir(asset_path) and os.path.exists(bins_csv):
            df = pd.read_csv(bins_csv)
            min_restriction = df["restriction_5th_percentile"].min()
            max_restriction = df["restriction_5th_percentile"].max()
            hp_range = df["hp_bin_center"].max() - df["hp_bin_center"].min()
            summary.append({
                "asset_id": asset_id,
                "min_restriction": min_restriction,
                "max_restriction": max_restriction,
                "hp_range": hp_range,
                "bins_csv": bins_csv,
                "plot_png": plot_png if os.path.exists(plot_png) else ""
            })
    return pd.DataFrame(summary)

def print_summary_table(df):
    print("Baseline Fit Summary Table")
    print(df[["asset_id", "min_restriction", "max_restriction", "hp_range"]].to_string(index=False))

def plot_comparison(df):
    plt.figure(figsize=(10, 6))
    for _, row in df.iterrows():
        if row["bins_csv"]:
            bins = pd.read_csv(row["bins_csv"])
            plt.plot(bins["hp_bin_center"], bins["restriction_5th_percentile"], label=row["asset_id"])
    plt.xlabel("HydraulicHorsepower")
    plt.ylabel("Restriction (5th percentile)")
    plt.title("Baseline Fit Comparison Across Assets")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_baseline_metrics()
    print_summary_table(df)
    plot_comparison(df)