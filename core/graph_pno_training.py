import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

# Configuration
DATASETS = ["usdcny", "coinbase", "audusd", "djia", "sp500", "usdjpy"]
# The caps you used in your training script
CAPS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0] 
OUTPUT_DIR = "graphs_pno_training"

def get_metrics(path):
    """Helper to load metrics from a result.json file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def process_dataset(dataset):
    print(f"Processing {dataset}...")
    
    # 1. Load the "Constraint-Aware" Results
    data = []
    for cap in CAPS:
        # Path format: output/PatchTST/usdcny_pno_0.1cap/result.json
        path = os.path.join("output", "PatchTST", f"{dataset}_pno_{cap}cap", "result.json")
        metrics = get_metrics(path)
        
        if metrics:
            data.append({
                "Cap_Size": cap,
                "Regret": metrics.get("Regret"),
                "Rel Regret": metrics.get("Rel Regret"),
                "MSE": metrics.get("MSE"),
                "MAE": metrics.get("MAE")
            })
    
    if not data:
        print(f"  No training data found for {dataset}. Skipping.")
        return

    df = pd.DataFrame(data)
    df = df.sort_values(by="Cap_Size")

    # 2. Load the "Original PnO" Baseline
    # Path format: output/PatchTST/usdcny_pno/result.json
    baseline_path = os.path.join("output", "PatchTST", f"{dataset}_pno", "result.json")
    baseline = get_metrics(baseline_path)
    
    if baseline:
        print(f"  Loaded Baseline (Original PnO)")
    else:
        print(f"  [Warning] No Baseline found at {baseline_path}")

    # 3. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Constraint-Aware Training: {dataset.upper()}', fontsize=16, fontweight='bold')

    metrics_list = ['Regret', 'Rel Regret', 'MSE', 'MAE']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'd']

    for i, ax in enumerate(axes.flat):
        metric = metrics_list[i]
        color = colors[i]
        
        # Plot the Curve (Constraint-Aware Models)
        if metric in df.columns:
            ax.plot(df['Cap_Size'], df[metric], marker=markers[i], color=color, 
                    linestyle='-', linewidth=2, markersize=8, label='Constraint-Aware PnO')
        
        # Plot the Baseline Line (Original PnO)
        if baseline and metric in baseline:
            val = baseline[metric]
            ax.axhline(y=val, color='gray', linestyle='--', linewidth=1.5, 
                       label=f'Original PnO ({val:.4f})')

        ax.set_title(f'{metric}', fontsize=12, pad=10)
        ax.set_xlabel('Training Cap Size')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 4. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{dataset}_training_sensitivity.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [Saved] {out_path}")

if __name__ == "__main__":
    for d in DATASETS:
        process_dataset(d)