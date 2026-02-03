import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

# Data from your latest report
RESULTS = {
    "USDBDT": {"imp": 48.04, "vol": 2.61},
    "USDSGD": {"imp": 15.92, "vol": 0.33},
    "USDCNY": {"imp": 11.18, "vol": 0.21},
    "USDRUB": {"imp": 10.49, "vol": 2.28},
    "USDTWD": {"imp": 10.34, "vol": 0.37},
    "AUDUSD": {"imp": 5.32,  "vol": 0.75},
    "USDJPY": {"imp": 4.45,  "vol": 1.31},
    "SP500":  {"imp": 0.37,  "vol": 21.16},
    "COINBASE": {"imp": -1.26, "vol": 31.01},
    "DJIA":     {"imp": -4.95, "vol": 15.76},
    "USDFJD":   {"imp": -8.99, "vol": 0.53},
}

DATASETS_FOR_SENSITIVITY = ["usdbdt", "coinbase", "usdcny"] # One huge win, one loss, one stable
CAPS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]

def plot_improvement_bar():
    """Generates a sorted bar chart of performance improvement."""
    df = pd.DataFrame.from_dict(RESULTS, orient='index').reset_index()
    df.columns = ['Dataset', 'Improvement', 'Volatility']
    df = df.sort_values(by='Improvement', ascending=False)

    colors = ['#2ca02c' if x > 0 else '#d62728' for x in df['Improvement']]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(df['Dataset'], df['Improvement'], color=colors, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        label_y = height + 1 if height > 0 else height - 2
        plt.text(bar.get_x() + bar.get_width()/2., label_y, f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("Regret Reduction (%)", fontsize=11)
    plt.title("Performance Improvement over Static Baseline", fontsize=13, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("paper_fig_improvement.png", dpi=300)
    print("Saved paper_fig_improvement.png")

def plot_sensitivity_lines():
    """Plots Regret vs Cap for select datasets to show contrast."""
    plt.figure(figsize=(8, 5))
    
    for dataset in DATASETS_FOR_SENSITIVITY:
        regrets = []
        caps = []
        
        # Get baseline for normalization
        base_path = f"output/PatchTST/{dataset}_pno_1.0cap/result.json"
        if not os.path.exists(base_path): continue
        with open(base_path, 'r') as f: baseline = json.load(f)['Regret']

        for cap in CAPS:
            path = f"output/PatchTST/{dataset}_mpc_{cap}cap/result.json"
            if os.path.exists(path):
                with open(path, 'r') as f:
                    val = json.load(f)['Regret']
                    # Normalize to show relative shape
                    regrets.append(val / baseline) 
                    caps.append(cap)
        
        if regrets:
            label = dataset.upper()
            if dataset == 'usdbdt': marker = 'o'; color='#2ca02c' # Green
            elif dataset == 'coinbase': marker = 'x'; color='#d62728' # Red
            else: marker = 's'; color='#1f77b4' # Blue
            
            plt.plot(caps, regrets, marker=marker, linewidth=2, label=label, color=color)

    plt.axhline(1.0, color='gray', linestyle=':', label="Static Baseline (1.0)")
    plt.xlabel("Training Safety Cap ($\kappa$)", fontsize=11)
    plt.ylabel("Normalized Regret (Lower is Better)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("paper_fig_sensitivity.png", dpi=300)
    print("Saved paper_fig_sensitivity.png")

if __name__ == "__main__":
    plot_improvement_bar()
    plot_sensitivity_lines()