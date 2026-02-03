import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_CONFIGS = {
    "usdcny":   0.75,
    "audusd":   0.25,
    "usdjpy":   0.7,
    "sp500":    0.9,
    "coinbase": 0.75,
    "djia":     0.7
}

BASELINE_CAP = 1.0 
OUTPUT_DIR = "graphs_standardized_impact"

# Exclusive Bins (Percentage of Average Cost)
# (Lower, Upper, Label)
BINS = [
    (0.01, 0.05, "1% - 5%"),
    (0.05, 0.10, "5% - 10%"),
    (0.10, 0.20, "10% - 20%"),
    (0.20, 0.50, "20% - 50%"),
    (0.50, 1.00, "50% - 100%"),   # Big Win
    (1.00, 2.00, "100% - 200%"),  # Massive Win (Saved 1x-2x Avg Cost)
    (2.00, float('inf'), "> 200%") # Extreme Win
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_json_lines(path):
    data = []
    if not os.path.exists(path): return None
    with open(path, 'r') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: continue
    return pd.DataFrame(data)

def plot_standardized_bars(dataset, df, save_path):
    # 1. Calculate Average Cost (The Stable Denominator)
    # We use the CONTINUOUS model's average cost as the benchmark for "Normal Day"
    avg_cont_regret = df['regret_cont'].mean()
    
    # 2. Calculate Standardized Impact
    # Positive = Saved X% of an average day
    # Negative = Lost X% of an average day
    # Avoid zero division
    denom = avg_cont_regret if avg_cont_regret != 0 else 1.0
    df['std_impact'] = (df['regret_std'] - df['regret_cont']) / denom

    win_counts = []
    loss_counts = []
    labels = []

    for low, high, label in BINS:
        labels.append(label)
        
        # Count Wins in range [low, high)
        n_wins = len(df[(df['std_impact'] >= low) & (df['std_impact'] < high)])
        win_counts.append(n_wins)
        
        # Count Losses in range (worse than -low, better than -high)
        # We use absolute value of the negative impact
        abs_loss_impact = df[df['std_impact'] < 0]['std_impact'].abs()
        n_losses = len(abs_loss_impact[(abs_loss_impact >= low) & (abs_loss_impact < high)])
        loss_counts.append(n_losses)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(14, 7)) # Wider figure for more bins
    
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, win_counts, width, label='Wins', color='#2ca02c', alpha=0.85)
    rects2 = ax.bar(x + width/2, loss_counts, width, label='Losses', color='#d62728', alpha=0.85)

    # Labels
    ax.set_xlabel('Impact Magnitude (% of Average Daily Regret)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Test Cases', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset.upper()} | Standardized Impact Distribution', fontsize=15, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Add Context Box (Average Value)
    stats_text = (
        f"CONTEXT:\n"
        f"Avg Daily Regret: {avg_cont_regret:.6f}\n"
        f"(This is the 100% baseline)"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Add Counts on Bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#444444')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  [Saved] {os.path.basename(save_path)}")

# ==========================================
# MAIN
# ==========================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"{'='*60}\n STANDARDIZED IMPACT ANALYSIS (Extended Range)\n{'='*60}")
    
    for dataset, best_cap in DATASET_CONFIGS.items():
        # Paths
        path_std_1 = os.path.join("output", "PatchTST", f"{dataset}_pno_{BASELINE_CAP}cap", "cases_test", "pno_cases.jsonl")
        path_std_2 = os.path.join("output", "PatchTST", f"{dataset}_pno", "cases_test", "pno_cases.jsonl")
        path_std = path_std_1 if os.path.exists(path_std_1) else path_std_2
        path_cont = os.path.join("output", "PatchTST", f"{dataset}_mpc_{best_cap}cap", "cases_test", "mpc_cases.jsonl")

        if not os.path.exists(path_std) or not os.path.exists(path_cont):
            print(f"  [Skip] {dataset}")
            continue

        df_std = load_json_lines(path_std).rename(columns={'regret': 'regret_std'})
        df_cont = load_json_lines(path_cont).rename(columns={'regret': 'regret_cont'})
        
        # Merge
        df = pd.merge(df_std[['case_id', 'regret_std']], df_cont[['case_id', 'regret_cont']], on='case_id')
        
        # Generate
        plot_standardized_bars(dataset, df, os.path.join(OUTPUT_DIR, f"{dataset}_std_impact.png"))

if __name__ == "__main__":
    run()