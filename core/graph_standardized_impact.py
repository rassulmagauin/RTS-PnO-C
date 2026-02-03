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
    "usdcny":   0.75
}

BASELINE_CAP = 1.0 
OUTPUT_DIR = "graphs_standardized_impact"

# Exclusive Bins (Percentage of Average Cost)
BINS = [
    (0.01, 0.05, "1% - 5%"),
    (0.05, 0.10, "5% - 10%"),
    (0.10, 0.20, "10% - 20%"),
    (0.20, 0.50, "20% - 50%"),
    (0.50, 1.00, "50% - 100%"),   
    (1.00, 2.00, "100% - 200%"),  
    (2.00, float('inf'), "> 200%") 
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
    # Calculate Standardized Impact
    # Denominator: We use the STATIC regret average as the baseline "scale"
    # This aligns with the definition: "Savings relative to baseline average cost"
    avg_static_regret = df['regret_std'].mean()
    denom = avg_static_regret if avg_static_regret != 0 else 1.0
    
    df['std_impact'] = (df['regret_std'] - df['regret_cont']) / denom

    win_counts = []
    loss_counts = []
    labels = []

    for low, high, label in BINS:
        labels.append(label)
        # Count Wins in this bin
        n_wins = len(df[(df['std_impact'] >= low) & (df['std_impact'] < high)])
        win_counts.append(n_wins)
        
        # Count Losses in this bin (using absolute value of negative impact)
        abs_loss_impact = df[df['std_impact'] < 0]['std_impact'].abs()
        n_losses = len(abs_loss_impact[(abs_loss_impact >= low) & (abs_loss_impact < high)])
        loss_counts.append(n_losses)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(14, 8)) # Slightly taller for better spacing
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, win_counts, width, label='Wins', color='#2ca02c', alpha=0.9)
    rects2 = ax.bar(x + width/2, loss_counts, width, label='Losses', color='#d62728', alpha=0.9)

    # --- INCREASED FONT SIZES HERE ---
    ax.set_xlabel('Impact Magnitude (% of Avg Static Regret)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Test Cases', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title(f'{dataset.upper()} | Standardized Impact Distribution', fontsize=20, fontweight='bold', pad=20)
    
    # Tick params
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=14) # Bigger ticks
    
    # Legend remains in top right
    ax.legend(fontsize=14, loc='upper right', framealpha=0.95, shadow=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # --- CONTEXT BOX (Top Left) ---
    stats_text = (
        f"CONTEXT:\n"
        f"Avg Static Regret: {avg_static_regret:.4f}\n"
        f"(>200% = Saving 2x Avg)"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='#444444', linewidth=1.5)
    
    ax.text(0.02, 0.96, stats_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='left', bbox=props, zorder=10)

    # --- ANNOTATIONS ---
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5),  # 5 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', 
                            fontsize=12, fontweight='bold', color='black') # Bigger numbers

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
    print(f"{'='*60}\n STANDARDIZED IMPACT ANALYSIS (Large Fonts)\n{'='*60}")
    
    for dataset, best_cap in DATASET_CONFIGS.items():
        path_std_1 = os.path.join("output", "PatchTST", f"{dataset}_pno_{BASELINE_CAP}cap", "cases_test", "pno_cases.jsonl")
        path_std_2 = os.path.join("output", "PatchTST", f"{dataset}_pno", "cases_test", "pno_cases.jsonl")
        path_std = path_std_1 if os.path.exists(path_std_1) else path_std_2
        path_cont = os.path.join("output", "PatchTST", f"{dataset}_mpc_{best_cap}cap", "cases_test", "mpc_cases.jsonl")

        if not os.path.exists(path_std) or not os.path.exists(path_cont):
            print(f"  [Skip] {dataset}")
            continue

        df_std = load_json_lines(path_std).rename(columns={'regret': 'regret_std'})
        df_cont = load_json_lines(path_cont).rename(columns={'regret': 'regret_cont'})
        
        # Merge only on common cases
        df = pd.merge(df_std[['case_id', 'regret_std']], df_cont[['case_id', 'regret_cont']], on='case_id')
        
        plot_standardized_bars(dataset, df, os.path.join(OUTPUT_DIR, f"{dataset}_std_impact.png"))

if __name__ == "__main__":
    run()