import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving only
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def process_and_save_graph(file_path):
    # 1. Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # 2. Load the data
    try:
        # sep='|' handles the pipe separators
        # skiprows=[1] ignores the '-------' line
        # skipinitialspace=True handles whitespace around data
        df = pd.read_csv(file_path, sep='|', skiprows=[1], skipinitialspace=True)
        
        # Clean whitespace from column names and string data
        df.columns = [c.strip() for c in df.columns]
        
        # Sort by Cap_Size to ensure the lines connect correctly
        df = df.sort_values(by='Cap_Size').reset_index(drop=True)
        
    except Exception as e:
        print(f"Error parsing the file: {e}")
        return

    # 3. Setup the visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Analysis for: {os.path.basename(file_path)}', fontsize=16, fontweight='bold')

    metrics = ['Abs_Regret', 'Rel_Regret', 'MSE', 'MAE']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'd']

    for i, ax in enumerate(axes.flat):
        metric = metrics[i]
        ax.plot(df['Cap_Size'], df[metric], marker=markers[i], color=colors[i], 
                linestyle='-', linewidth=2, markersize=8)
        ax.set_title(f'{metric} vs Cap_Size', fontsize=12, pad=10)
        ax.set_xlabel('Cap_Size')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 4. Generate the output filename
    # os.path.splitext splits "data.txt" into ("data", ".txt")
    base_name, _ = os.path.splitext(file_path)
    output_filename = f"{base_name}_graph.png"

    # 5. Save and Close
    plt.savefig(output_filename, dpi=300) # dpi=300 for high quality
    plt.close(fig) 
    print(f"Successfully saved graph to: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_and_save_graph(sys.argv[1])
    else:
        path = input("Please enter the path to your .txt file: ")
        process_and_save_graph(path)