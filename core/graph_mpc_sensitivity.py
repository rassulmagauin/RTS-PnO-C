import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

# Configuration
DATASETS = ["usdrub", "usdtwd", "usdbdt", "usdfjd", "usdsgd"] 
CAPS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
OUTPUT_DIR = "graphs_mpc_sensitivity"

def get_metrics(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def get_static_baseline(dataset, metric_name):
    """
    Fetches the metric from the Static PnO model (One-Time Allocation).
    Prioritizes '_pno_1.0cap', falls back to '_pno'.
    """
    path = os.path.join("output", "PatchTST", f"{dataset}_pno_1.0cap", "result.json")
    
    if not os.path.exists(path):
        path = os.path.join("output", "PatchTST", f"{dataset}_pno", "result.json")

    metrics = get_metrics(path)
    if metrics and metric_name in metrics:
        return metrics[metric_name]
    return None

def print_comparison_table(dataset, df, baseline_regret):
    """Prints a formatted text table to the terminal."""
    print(f"\n{'='*60}")
    print(f" RESULTS: {dataset.upper()}")
    print(f"{'='*60}")
    
    # Find Best Continuous Result
    if not df.empty:
        best_row = df.loc[df['Regret'].idxmin()]
        best_cap = best_row['Cap_Size']
        best_regret = best_row['Regret']
    else:
        best_regret = float('inf')

    # Header
    print(f"{'Model Type':<25} | {'Cap Size':<10} | {'Regret (Cost)':<15} | {'Improvement':<12}")
    print("-" * 70)

    # 1. Print Static Baseline
    base_str = f"{baseline_regret:.6f}" if baseline_regret else "MISSING"
    print(f"{'Static PnO (Baseline)':<25} | {'1.0':<10} | {base_str:<15} | {'0.00%':<12}")

    # 2. Print Continuous Results
    if not df.empty:
        for _, row in df.iterrows():
            cap = row['Cap_Size']
            regret = row['Regret']
            
            # Calculate Improvement vs Baseline
            if baseline_regret:
                diff_pct = ((baseline_regret - regret) / baseline_regret) * 100
                diff_str = f"{diff_pct:+.2f}%"
                
                # Visual marker for the winner
                marker = " << BEST" if cap == best_cap else ""
                if diff_pct > 0:
                    marker = " (Win)" if cap != best_cap else " << BEST WIN"
                
                print(f"{'Continuous RTS-PnO':<25} | {cap:<10} | {regret:.6f}        | {diff_str:<12}{marker}")
            else:
                print(f"{'Continuous RTS-PnO':<25} | {cap:<10} | {regret:.6f}        | {'N/A':<12}")

    print("-" * 70)
    if baseline_regret and best_regret < float('inf'):
        total_imp = ((baseline_regret - best_regret) / baseline_regret) * 100
        if total_imp > 0:
            print(f"✅ SUCCESS: Continuous System reduced regret by {total_imp:.2f}% (Cap {best_cap})")
        else:
            print(f"❌ RESULT: Static Baseline was better by {abs(total_imp):.2f}%")
    print("\n")

def process_dataset(dataset):
    data = []
    for cap in CAPS:
        path = os.path.join("output", "PatchTST", f"{dataset}_mpc_{cap}cap", "result.json")
        metrics = get_metrics(path)
        
        if metrics:
            data.append({
                "Cap_Size": cap,
                "Regret": metrics.get("Regret"),
                "Rel Regret": metrics.get("Rel Regret"),
                "MSE": metrics.get("MSE", 0),
                "MAE": metrics.get("MAE", 0)
            })
    
    if not data:
        print(f"  [Skip] No Continuous data found for {dataset}.")
        return

    df = pd.DataFrame(data)
    df = df.sort_values(by="Cap_Size")
    
    # --- PRINT TEXT RESULTS ---
    baseline_regret = get_static_baseline(dataset, 'Regret')
    print_comparison_table(dataset, df, baseline_regret)

    # --- GENERATE GRAPH ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Continuous RTS-PnO Performance: {dataset.upper()}', fontsize=16, fontweight='bold')

    metrics_list = ['Regret', 'Rel Regret', 'MSE', 'MAE']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'd']

    for i, ax in enumerate(axes.flat):
        metric = metrics_list[i]
        color = colors[i]
        
        if metric in df.columns:
            # Plot Continuous Curve
            ax.plot(df['Cap_Size'], df[metric], marker=markers[i], color=color, 
                    linestyle='-', linewidth=2, markersize=8, label='Continuous RTS-PnO')
            
            # Plot Static Baseline Line
            baseline_val = get_static_baseline(dataset, metric)
            if baseline_val is not None:
                ax.axhline(y=baseline_val, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, 
                           label=f'Static PnO (Cap 1.0)')

        ax.set_title(f'{metric}', fontsize=12, pad=10)
        ax.set_xlabel('Constraint Cap Size')
        ax.set_ylabel('Value (Lower is Better)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{dataset}_continuous_sensitivity.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    for d in DATASETS:
        process_dataset(d)