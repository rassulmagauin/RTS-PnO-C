import os
import json
import pandas as pd

# Define your datasets and caps
DATASETS = ["audusd", "usdcny", "coinbase", "djia", "sp500", "usdjpy"]
CAPS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]

def collect_and_save(dataset):
    results = []
    print(f"\nProcessing {dataset}...")
    
    for cap in CAPS:
        # Construct path to the result file
        exp_id = f"PatchTST/{dataset}_mpc_{cap}cap"
        file_path = os.path.join("output", exp_id, "result.json")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Append to list
                results.append({
                    "Cap_Size": cap,
                    "Abs_Regret": data.get("Regret", "N/A"),
                    "Rel_Regret": data.get("Rel Regret", "N/A"),
                    "MSE": data.get("MSE", "N/A"),
                    "MAE": data.get("MAE", "N/A")
                })
            except Exception as e:
                print(f"  Error reading cap {cap}: {e}")
        else:
            # print(f"  Missing: {file_path}") # Uncomment to debug missing files
            pass

    if not results:
        print(f"  No results found for {dataset}.")
        return

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by Cap Size
    df = df.sort_values(by="Cap_Size")

    # Save to text file (matching your old format)
    output_file = f"summary_{dataset}.txt"
    with open(output_file, "w") as f:
        f.write("Cap_Size | Abs_Regret | Rel_Regret | MSE | MAE\n")
        f.write("-" * 57 + "\n")
        for _, row in df.iterrows():
            f.write(f"{row['Cap_Size']:<8} | {row['Abs_Regret']} | {row['Rel_Regret']} | {row['MSE']} | {row['MAE']}\n")
    
    print(f"  [Saved] {output_file}")
    
    # Optional: Also print to terminal
    print(df.to_string(index=False))

if __name__ == "__main__":
    for d in DATASETS:
        collect_and_save(d)