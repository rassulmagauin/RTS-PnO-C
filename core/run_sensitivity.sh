#!/bin/bash

# ==============================================================================
# FAST SENSITIVITY ANALYSIS (MPC ONLY)
# ==============================================================================

# 1. Define Datasets (Excluding usdcny as you already have it)
DATASETS=("usdcny")

# 2. Define Budget Caps to test
CAPS=(0.1 0.25 0.5 0.75 1.0)

PYTHON_EXEC="python"

# Helper to generate config
generate_yaml() {
    local dataset=$1
    local cap=$2
    local output_file=$3
    
    $PYTHON_EXEC -c "
import yaml
import os

pno_config_path = 'configs/PatchTST/${dataset}/pno.yaml'
with open(pno_config_path, 'r') as f:
    config = yaml.safe_load(f)

# Modify for MPC
config['Experiment']['exp_id'] = f'PatchTST/${dataset}_mpc_${cap}cap'
config['Experiment']['prev_exp_id'] = f'PatchTST/${dataset}_pno'
config['Experiment']['mpc_first_step_cap'] = float(${cap})

# Ensure critical hardware/allocation settings
if 'Hardware' not in config: config['Hardware'] = {}
config['Hardware']['num_workers'] = 0 
if 'Allocate' not in config: config['Allocate'] = {}
config['Allocate']['batch_size'] = 32
config['Allocate']['uncertainty_quantile'] = 0.3 

with open('${output_file}', 'w') as f:
    yaml.dump(config, f, sort_keys=False)
"
}

# Helper to extract metrics
extract_metric() {
    local json_file=$1
    local key=$2
    if [ -f "$json_file" ]; then
        $PYTHON_EXEC -c "import json; print(json.load(open('$json_file'))['$key'])" 2>/dev/null
    else
        echo "N/A"
    fi
}

# ==============================================================================
# MAIN LOOP
# ==============================================================================

for DATASET in "${DATASETS[@]}"; do
    echo "=================================================================="
    echo "PROCESSING DATASET: $DATASET"
    echo "=================================================================="
    
    SUMMARY_FILE="summary_${DATASET}.txt"
    # Create header if file doesn't exist
    if [ ! -f "$SUMMARY_FILE" ]; then
        echo "Cap_Size | Abs_Regret | Rel_Regret | MSE | MAE" > "$SUMMARY_FILE"
        echo "---------------------------------------------------------" >> "$SUMMARY_FILE"
    fi

    # Check if PnO model exists (Safety Check)
    PNO_MODEL="output/PatchTST/${DATASET}_pno/model.pt"
    if [ ! -f "$PNO_MODEL" ]; then
        echo "[ERROR] PnO model not found for $DATASET! Skipping..."
        continue
    fi

    # Run MPC Loops
    for CAP in "${CAPS[@]}"; do
        echo "  > Running MPC with Cap: $CAP"
        
        TEMP_CONFIG="configs/PatchTST/${DATASET}/temp_mpc_${CAP}.yaml"
        generate_yaml "$DATASET" "$CAP" "$TEMP_CONFIG"
        
        # Run MPC
        $PYTHON_EXEC src/run_mpc.py --config "$TEMP_CONFIG"
        
        # Collect Results
        RESULT_JSON="output/PatchTST/${DATASET}_mpc_${CAP}cap/result.json"
        
        if [ -f "$RESULT_JSON" ]; then
            REGRET=$(extract_metric "$RESULT_JSON" "Regret")
            REL_REG=$(extract_metric "$RESULT_JSON" "Rel Regret")
            MSE=$(extract_metric "$RESULT_JSON" "MSE")
            MAE=$(extract_metric "$RESULT_JSON" "MAE")
            
            # Format nicely
            printf "%-8s | %-10s | %-10s | %-5s | %-5s\n" "$CAP" "$REGRET" "$REL_REG" "$MSE" "$MAE" >> "$SUMMARY_FILE"
            echo "    [DONE] Regret: $REGRET"
        else
            echo "    [FAIL] No result.json found!"
        fi
        
        rm "$TEMP_CONFIG"
    done
    
    echo "Finished $DATASET. Results in $SUMMARY_FILE"
    echo ""
done