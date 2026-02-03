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
OUTPUT_DIR = "graphs_win_magnitude"

# Exclusive Bins (Lower Inclusive, Upper Exclusive)
# Format: (Lower_Bound, Upper_Bound, Label)
BINS = [
    (0.001, 0.01, "0.1% - 1%"),   # Marginal
    (0.01,  0.05, "1% - 5%"),     # Small
    (0.05,  0.10, "5% - 10%"),    # Significant
    (0.10,  0.20, "10% - 20%"),   # Major
    (0.20,  0.50, "20% - 50%"),   # Huge
    (0.50,  float('inf'), "> 50%") # Extreme / Crisis
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_json_lines(path):
    data = []
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: continue
    return pd.DataFrame(data)

def plot_exclusive_bars(dataset, df, save_path):
    """
    Plots grouped bars for Wins vs Losses using exclusive bins.
    """
    # Calculate % Difference
    df = df[df['regret_std'] != 0].copy()
    
    # Positive = Win (Saved Money)
    # Negative = Loss (Cost Extra)
    df['diff_pct'] = (df['regret_std'] - df['regret_cont']) / df['regret_std']

    win_counts = []
    loss_counts = []
    labels = []

    for low, high, label in BINS:
        labels.append(label)
        
        # Count Wins in range [low, high)
        n_wins = len(df[(df['diff_pct'] >= low) & (df['diff_pct'] < high)])
        win_counts.append(n_wins)
        
        # Count Losses in range (worse than -low, but better than -high)
        # Logic: -high < diff <= -low
        # We take absolute value for the range check
        abs_loss = df[df['diff_pct'] < 0]['diff_pct'].abs()
        n_losses = len(abs_loss[(abs_loss >= low) & (abs_loss < high)])
        loss_counts.append(n_losses)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(labels))
    width = 0.35

    # Draw Bars
    rects1 = ax.bar(x - width/2, win_counts, width, label='Wins', color='#2ca02c', alpha=0.85)
    rects2 = ax.bar(x + width/2, loss_counts, width, label='Losses', color='#d62728', alpha=0.85)

    # Labels
    ax.set_xlabel('Magnitude Range (% Improvement over Baseline)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Test Cases', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset.upper()} | Win/Loss Distribution (Exclusive Bins)', fontsize=15, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444444')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  [Saved] {os.path.basename(save_path)}")

# ==========================================
# MAIN LOGIC
# ==========================================
def run_analysis():
    print(f"{'='*60}")
    print(f" EXCLUSIVE BIN ANALYSIS")
    print(f"{'='*60}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset, best_cap in DATASET_CONFIGS.items():
        print(f"Processing: {dataset.upper()}...")
        
        path_std_1 = os.path.join("output", "PatchTST", f"{dataset}_pno_{BASELINE_CAP}cap", "cases_test", "pno_cases.jsonl")
        path_std_2 = os.path.join("output", "PatchTST", f"{dataset}_pno", "cases_test", "pno_cases.jsonl")
        path_std = path_std_1 if os.path.exists(path_std_1) else path_std_2
        
        path_cont = os.path.join("output", "PatchTST", f"{dataset}_mpc_{best_cap}cap", "cases_test", "mpc_cases.jsonl")

        if not os.path.exists(path_std) or not os.path.exists(path_cont):
            print(f"  [Skip] Logs missing for {dataset}")
            continue

        df_std = load_json_lines(path_std)
        df_cont = load_json_lines(path_cont)
        
        df_std = df_std.rename(columns={'regret': 'regret_std'})
        df_cont = df_cont.rename(columns={'regret': 'regret_cont'})
        
        df = pd.merge(df_std[['case_id', 'regret_std']], df_cont[['case_id', 'regret_cont']], on='case_id')
        
        plot_exclusive_bars(dataset, df, os.path.join(OUTPUT_DIR, f"{dataset}_exclusive_dist.png"))

if __name__ == "__main__":
    run_analysis()