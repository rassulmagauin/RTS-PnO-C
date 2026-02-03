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
# Map each dataset to its BEST performing Cap size
DATASET_CONFIGS = {
    "usdcny":   0.75,
    "audusd":   0.25,
    "usdjpy":   0.7,
    "sp500":    0.9,
    "coinbase": 0.75, 
    "djia":     0.7   
}

BASELINE_CAP = 1.0 
OUTPUT_BASE_DIR = "graphs_detailed_cases"

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

def plot_deep_dive(row, title_prefix, filename, dataset_name, cap_size):
    """
    Generates a rich visualization comprising Price, Predictions, and Allocations.
    """
    # Data Extraction
    true_prices = np.array(row['true_prices_std'])
    pred_static = np.array(row['pred_prices_std']) if 'pred_prices_std' in row else None
    pred_cont   = np.array(row['pred_prices_cont']) if 'pred_prices_cont' in row else None
    alloc_static = np.array(row['alloc_std'])
    alloc_cont   = np.array(row['alloc_cont'])
    
    # Sync lengths
    min_len = len(true_prices)
    if pred_static is not None: min_len = min(min_len, len(pred_static))
    if pred_cont is not None: min_len = min(min_len, len(pred_cont))
    min_len = min(min_len, len(alloc_static), len(alloc_cont))
    
    # Trim data
    steps = np.arange(min_len)
    true_prices = true_prices[:min_len]
    if pred_static is not None: pred_static = pred_static[:min_len]
    if pred_cont is not None: pred_cont = pred_cont[:min_len]
    alloc_static = alloc_static[:min_len]
    alloc_cont = alloc_cont[:min_len]

    # --- PLOT SETUP ---
    fig, ax1 = plt.subplots(figsize=(15, 9)) # Slightly larger figure
    
    # 1. PRICES (Left Axis)
    ax1.set_xlabel('Time Step', fontsize=18, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Asset Price', color='black', fontsize=18, fontweight='bold', labelpad=10)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=14) # Bigger ticks
    
    # Ground Truth
    ax1.plot(steps, true_prices, color='black', linewidth=3.0, label='Ground Truth', zorder=10)
    
    # Predictions
    if pred_static is not None:
        ax1.plot(steps, pred_static, color='#d62728', linestyle='--', linewidth=2.0, alpha=0.8, label='Static Forecast (t=0)')
    if pred_cont is not None:
        ax1.plot(steps, pred_cont, color='#2ca02c', linestyle=':', linewidth=2.5, alpha=0.9, label='Rolling Forecast (MPC)')

    # 2. ALLOCATIONS (Right Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Allocation Amount', color='#555555', fontsize=18, fontweight='bold', labelpad=10)
    ax2.set_ylim(0, 1.15) # Slight increase to prevent bar overlap with legend
    ax2.tick_params(axis='y', labelsize=14, labelcolor='#555555')
    
    # Bar Plots
    width = 0.35
    ax2.bar(steps - width/2, alloc_static, width, color='#d62728', alpha=0.3, label='Static Buy (Baseline)')
    ax2.bar(steps + width/2, alloc_cont, width, color='#2ca02c', alpha=0.5, label='Continuous Buy (Ours)')

    # --- FINAL POLISH ---
    # Create unified legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    # Increased Legend Font
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', 
               frameon=True, fancybox=True, shadow=True, ncol=2, fontsize=14)
    
    # Stats Box
    saved_money = row['regret_std'] - row['regret_cont']
    imp_pct = (saved_money / row['regret_std']) * 100 if row['regret_std'] != 0 else 0
    
    stats_text = (
        f"Case #{row['case_id']}\n"
        f"Saved: {saved_money:.4f} ({imp_pct:+.1f}%)\n"
        f"Static Cost: {row['regret_std']:.4f}\n"
        f"Ours Cost:   {row['regret_cont']:.4f}"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='#333333')
    # Increased Stats Box Font
    ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, fontsize=14,
             verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=100)

    # Increased Title Font
    plt.title(f"{title_prefix} | {dataset_name.upper()} | Best Cap {cap_size}", fontsize=22, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"    -> [Saved] {os.path.basename(filename)}")

# ==========================================
# MAIN LOGIC
# ==========================================
def run_analysis():
    print(f"{'='*60}")
    print(f" BATCH VISUALIZATION & WIN-RATE ANALYSIS (Large Fonts)")
    print(f"{'='*60}")
    
    results_summary = []

    for dataset, best_cap in DATASET_CONFIGS.items():
        print(f"\nProcessing: {dataset.upper()} (Best Cap: {best_cap})...")
        
        # Paths
        path_std_1 = os.path.join("output", "PatchTST", f"{dataset}_pno_{BASELINE_CAP}cap", "cases_test", "pno_cases.jsonl")
        path_std_2 = os.path.join("output", "PatchTST", f"{dataset}_pno", "cases_test", "pno_cases.jsonl")
        path_std = path_std_1 if os.path.exists(path_std_1) else path_std_2
        
        path_cont = os.path.join("output", "PatchTST", f"{dataset}_mpc_{best_cap}cap", "cases_test", "mpc_cases.jsonl")

        if not os.path.exists(path_std) or not os.path.exists(path_cont):
            print(f"  [Skip] Logs not found for {dataset}.")
            continue

        # Load & Merge
        df_std = load_json_lines(path_std)
        df_cont = load_json_lines(path_cont)
        
        # Rename columns
        df_std = df_std.rename(columns={'regret': 'regret_std', 'alloc': 'alloc_std', 
                                        'true_prices': 'true_prices_std', 'pred_prices': 'pred_prices_std'})
        df_cont = df_cont.rename(columns={'regret': 'regret_cont', 'alloc': 'alloc_cont', 
                                          'true_prices': 'true_prices_cont', 'pred_prices': 'pred_prices_cont'})
        
        # Merge
        df = pd.merge(df_std, df_cont, on='case_id')
        
        # --- CALCULATE STATISTICS ---
        # "Win" = Continuous Regret < Static Regret
        df['diff'] = df['regret_std'] - df['regret_cont']
        
        total_cases = len(df)
        wins_count = len(df[df['diff'] > 0])
        draws_count = len(df[df['diff'] == 0])
        losses_count = len(df[df['diff'] < 0])
        
        win_rate = (wins_count / total_cases) * 100
        
        print(f"  > Total Cases: {total_cases}")
        print(f"  > Wins: {wins_count} ({win_rate:.2f}%)")
        print(f"  > Losses: {losses_count}")
        
        results_summary.append({
            "Dataset": dataset,
            "Best Cap": best_cap,
            "Win Rate": f"{win_rate:.2f}%"
        })

        # --- GENERATE GRAPHS ---
        output_dir = os.path.join(OUTPUT_BASE_DIR, dataset)
        os.makedirs(output_dir, exist_ok=True)
        
        wins = df[df['diff'] > 0].sort_values(by='diff')
        losses = df[df['diff'] < 0].sort_values(by='diff')

        if not wins.empty:
            # Strong Win (60th Percentile - Good but not outlier)
            idx = int(len(wins) * 0.60)
            plot_deep_dive(wins.iloc[idx], "Strong Win", os.path.join(output_dir, "Case_Win_Strong.png"), dataset, best_cap)
            
            # Median Win
            idx = int(len(wins) * 0.50)
            plot_deep_dive(wins.iloc[idx], "Typical Win", os.path.join(output_dir, "Case_Win_Median.png"), dataset, best_cap)

        if not losses.empty:
            # Median Loss
            idx = int(len(losses) * 0.50)
            plot_deep_dive(losses.iloc[idx], "Typical Loss", os.path.join(output_dir, "Case_Loss_Median.png"), dataset, best_cap)

    # --- FINAL TABLE PRINT ---
    print(f"\n{'='*60}")
    print(f"{'Dataset':<15} | {'Best Cap':<10} | {'Win Rate (vs Static)':<20}")
    print("-" * 60)
    for res in results_summary:
        print(f"{res['Dataset']:<15} | {str(res['Best Cap']):<10} | {res['Win Rate']:<20}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_analysis()