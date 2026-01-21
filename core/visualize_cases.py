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
DATASET = "usdcny"
BEST_CAP = 0.75       # Your best continuous cap
BASELINE_CAP = 1.0    # Static baseline
OUTPUT_DIR = f"graphs_cases_representative/{DATASET}"

# ==========================================
# HELPERS
# ==========================================
def load_json_lines(path):
    data = []
    if not os.path.exists(path):
        print(f"[Error] Missing: {path}")
        return None
    with open(path, 'r') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: continue
    return pd.DataFrame(data)

def plot_combined_case(row, title_prefix, filename):
    """
    Plots Price vs Allocation on a single chart for clarity.
    """
    prices = row['true_prices_std']
    alloc_static = row['alloc_std']
    alloc_cont = row['alloc_cont']
    
    # Sync lengths
    min_len = min(len(prices), len(alloc_static), len(alloc_cont))
    prices = prices[:min_len]
    alloc_static = alloc_static[:min_len]
    alloc_cont = alloc_cont[:min_len]
    steps = np.arange(len(prices))

    # Create Figure
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # --- PLOT 1: PRICE (Line) ---
    color_price = '#333333'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Asset Price', color=color_price, fontsize=12)
    ax1.plot(steps, prices, color=color_price, linewidth=2.5, label='True Price', zorder=10)
    ax1.tick_params(axis='y', labelcolor=color_price)
    
    # Highlight Min Price
    min_price_idx = np.argmin(prices)
    ax1.scatter(min_price_idx, prices[min_price_idx], color='gold', edgecolor='black', s=100, zorder=11, label="Lowest Price")

    # --- PLOT 2: ALLOCATIONS (Bars) ---
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Allocation Amount (Fraction of Budget)', color='#555555', fontsize=12)
    
    # Offset bars so they don't overlap
    width = 0.35
    ax2.bar(steps - width/2, alloc_static, width, color='#d62728', alpha=0.6, label=f'Static (Cap 1.0)')
    ax2.bar(steps + width/2, alloc_cont, width, color='#2ca02c', alpha=0.7, label=f'Continuous (Cap {BEST_CAP})')
    
    # Limits & Style
    ax2.set_ylim(0, 1.1)
    ax2.spines['top'].set_visible(False)
    
    # Combined Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, fancybox=True, shadow=True)

    # Title with Stats
    diff = row['diff']
    res_type = "WIN" if diff > 0 else "LOSS"
    plt.title(f"{title_prefix} | Case #{row['case_id']} | {res_type} (Saved: {diff:.2f})", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"  [Saved] {filename}")

# ==========================================
# MAIN
# ==========================================
def run_analysis():
    print(f"Finding representative cases for {DATASET}...")
    
    # Paths
    base_path_1 = os.path.join("output", "PatchTST", f"{DATASET}_pno_{BASELINE_CAP}cap", "cases_test", "pno_cases.jsonl")
    base_path_2 = os.path.join("output", "PatchTST", f"{DATASET}_pno", "cases_test", "pno_cases.jsonl")
    path_std = base_path_1 if os.path.exists(base_path_1) else base_path_2
    
    path_cont = os.path.join("output", "PatchTST", f"{DATASET}_mpc_{BEST_CAP}cap", "cases_test", "mpc_cases.jsonl")

    # Load & Merge
    df_std = load_json_lines(path_std)
    df_cont = load_json_lines(path_cont)
    if df_std is None or df_cont is None: return

    df_std = df_std.rename(columns={'regret': 'regret_std', 'alloc': 'alloc_std', 'true_prices': 'true_prices_std'})
    df_cont = df_cont.rename(columns={'regret': 'regret_cont', 'alloc': 'alloc_cont'})
    
    df = pd.merge(df_std[['case_id', 'regret_std', 'alloc_std', 'true_prices_std']], 
                  df_cont[['case_id', 'regret_cont', 'alloc_cont']], on='case_id')
    
    # Calculate Savings (Positive = Good)
    df['diff'] = df['regret_std'] - df['regret_cont']
    
    # Separate Wins and Losses
    wins = df[df['diff'] > 0].sort_values(by='diff')
    losses = df[df['diff'] < 0].sort_values(by='diff')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- SELECTION LOGIC: "Representative" means not the extreme tail ---
    
    # 1. Strong Win (90th Percentile) - Shows high performance, but filters likely glitches
    if not wins.empty:
        idx_90 = int(len(wins) * 0.90)
        row = wins.iloc[idx_90]
        plot_combined_case(row, "Strong Win (90th %)", os.path.join(OUTPUT_DIR, "Win_Strong_Representative.png"))

    # 2. Typical Win (Median) - Shows the "average day" improvement
    if not wins.empty:
        idx_50 = int(len(wins) * 0.50)
        row = wins.iloc[idx_50]
        plot_combined_case(row, "Typical Win (Median)", os.path.join(OUTPUT_DIR, "Win_Typical_Median.png"))

    # 3. Typical Loss (Median) - Shows where it normally struggles
    if not losses.empty:
        idx_50 = int(len(losses) * 0.50)
        row = losses.iloc[idx_50]
        plot_combined_case(row, "Typical Loss (Median)", os.path.join(OUTPUT_DIR, "Loss_Typical_Median.png"))

if __name__ == "__main__":
    run_analysis()