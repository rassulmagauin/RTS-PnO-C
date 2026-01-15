#!/bin/bash

# ==============================================================================
# PARALLEL SENSITIVITY ANALYSIS (Utilization: 100%)
# ==============================================================================

# 1. Define Datasets and Caps
DATASETS=("coinbase" "usdcny")
CAPS=(0.1 0.25 0.5 0.75 1.0)
PYTHON_EXEC="python"

# Create logs directory to keep things clean
mkdir -p logs_sensitivity

echo "=================================================================="
echo "Phase 1: Generating Config Files"
echo "=================================================================="

# Python snippet to generate config
generate_yaml() {
    local dataset=$1
    local cap=$2
    local output_file=$3
    
    $PYTHON_EXEC -c "
import yaml
import os

# --- FIX: Define the output filename as a Python string first ---
target_file = '${output_file}'
pno_config_path = 'configs/PatchTST/${dataset}/pno.yaml'

try:
    with open(pno_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for MPC
    config['Experiment']['exp_id'] = f'PatchTST/${dataset}_mpc_${cap}cap'
    config['Experiment']['prev_exp_id'] = f'PatchTST/${dataset}_pno'
    config['Experiment']['mpc_first_step_cap'] = float(${cap})

    # Optimization settings
    if 'Hardware' not in config: config['Hardware'] = {}
    config['Hardware']['num_workers'] = 0 
    if 'Allocate' not in config: config['Allocate'] = {}
    config['Allocate']['batch_size'] = 32
    config['Allocate']['uncertainty_quantile'] = 0.3 

    # Write to the target file variable
    with open(target_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        
    print(f'Generated: {target_file}')

except FileNotFoundError:
    print(f'[SKIP] Config not found: {pno_config_path}')
except Exception as e:
    print(f'[ERROR] Failed to generate {target_file}: {e}')
"
}

# Loop to generate ALL configs upfront
for DATASET in "${DATASETS[@]}"; do
    for CAP in "${CAPS[@]}"; do
        CONFIG_FILE="configs/PatchTST/${DATASET}/temp_mpc_${CAP}.yaml"
        generate_yaml "$DATASET" "$CAP" "$CONFIG_FILE"
    done
done

echo ""
echo "=================================================================="
echo "Phase 2: Launching Experiments in Parallel"
echo "=================================================================="

# Launch ALL experiments in the background
for DATASET in "${DATASETS[@]}"; do
    # Safety Check: PnO model must exist
    PNO_MODEL="output/PatchTST/${DATASET}_pno/model.pt"
    if [ ! -f "$PNO_MODEL" ]; then
        echo "[WARNING] PnO model not found for $DATASET! Skipping."
        continue
    fi

    for CAP in "${CAPS[@]}"; do
        CONFIG_FILE="configs/PatchTST/${DATASET}/temp_mpc_${CAP}.yaml"
        LOG_FILE="logs_sensitivity/${DATASET}_mpc_${CAP}.log"
        
        if [ -f "$CONFIG_FILE" ]; then
            echo " > Launching: $DATASET (Cap $CAP) -> Logs: $LOG_FILE"
            # The '&' puts it in background immediately
            nohup $PYTHON_EXEC src/run_mpc.py --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
        else
            echo " > [SKIPPING] Config missing: $CONFIG_FILE"
        fi
    done
done

echo ""
echo "=================================================================="
echo "Phase 3: Waiting for completion..."
echo "=================================================================="
echo "All jobs are running. Check 'htop' to see 100% CPU usage."
echo "The script will pause here until ALL experiments are finished."

# 'wait' pauses this script until all background jobs started by it are done
wait

echo ""
echo "=================================================================="
echo "Phase 4: Cleanup"
echo "=================================================================="

# Delete the temp configs
rm configs/PatchTST/*/temp_mpc_*.yaml
echo "Cleaned up temporary config files."
echo "Done."