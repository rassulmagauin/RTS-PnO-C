import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving only
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import json

def get_pno_metrics(dataset_name):
    """
    Attempts to load PnO results for the given dataset.
    Assumes standard directory structure: output/PatchTST/{dataset}_pno/result.json
    """
    # Try to construct the path to the PnO result file
    # We assume 'PatchTST' because that is what collect_results.py uses
    pno_path = os.path.join("output", "PatchTST", f"{dataset_name}_pno", "result.json")
    
    if not os.path.exists(pno_path):
        print(f"   [Info] PnO baseline file not found at: {pno_path}")
        print(f"   (Graph will be generated without PnO line)")
        return None
        
    try:
        with open(pno_path, 'r') as f:
            data = json.load(f)
        print(f"   [Info] Loaded PnO baseline from: {pno_path}")
        return data
    except Exception as e:
        print(f"   [Warning] Could not read PnO file: {e}")
        return None

def process_and_save_graph(file_path):
    # 1. Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # 2. Load the MPC data
    try:
        df = pd.read_csv(file_path, sep='|', skiprows=[1], skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        df = df.sort_values(by='Cap_Size').reset_index(drop=True)
    except Exception as e:
        print(f"Error parsing the file: {e}")
        return

    # 3. Try to get PnO Data
    # Extract dataset name from filename (e.g. "summary_usdcny.txt" -> "usdcny")
    filename = os.path.basename(file_path)
    dataset_name = filename.replace("summary_", "").replace(".txt", "")
    pno_metrics = get_pno_metrics(dataset_name)

    # Map text file columns to JSON keys
    metric_map = {
        'Abs_Regret': 'Regret',
        'Rel_Regret': 'Rel Regret',
        'MSE': 'MSE',
        'MAE': 'MAE'
    }

    # 4. Setup the visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MPC Analysis: {dataset_name.upper()}', fontsize=16, fontweight='bold')

    metrics = ['Abs_Regret', 'Rel_Regret', 'MSE', 'MAE']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'd']

    for i, ax in enumerate(axes.flat):
        metric = metrics[i]
        
        # Plot MPC Line
        ax.plot(df['Cap_Size'], df[metric], marker=markers[i], color=colors[i], 
                linestyle='-', linewidth=2, markersize=8, label='MPC')
        
        # Plot PnO Baseline (if available)
        if pno_metrics:
            pno_key = metric_map.get(metric)
            if pno_key in pno_metrics:
                val = pno_metrics[pno_key]
                ax.axhline(y=val, color='gray', linestyle='--', linewidth=1.5, label=f'PnO ({val:.4f})')

        ax.set_title(f'{metric}', fontsize=12, pad=10)
        ax.set_xlabel('Cap_Size')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend() # Add legend to show MPC vs PnO

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 5. Generate output filename
    base_name, _ = os.path.splitext(file_path)
    output_filename = f"{base_name}_graph.png"

    # 6. Save
    plt.savefig(output_filename, dpi=300)
    plt.close(fig) 
    print(f"Successfully saved graph to: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_and_save_graph(sys.argv[1])
    else:
        path = input("Please enter the path to your .txt file: ")
        process_and_save_graph(path)