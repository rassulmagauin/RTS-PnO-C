#!/bin/bash

# ==============================================================================
# BATCHED SENSITIVITY ANALYSIS (Safe for GPU & CPU)
# ==============================================================================

# 1. Define Datasets and Caps
DATASETS=("audusd" "usdcny" "coinbase" "djia" "sp500" "usdjpy")
CAPS=(0.1 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.9 1.0)
PYTHON_EXEC="python"

# --- MAX PARALLEL JOBS ---
# We limit to 10 concurrent experiments.
# 10 Exps * 4 Internal Threads = 40 Threads (Fits on 48-core CPU)
# 10 Models on GPU = ~10-15GB VRAM (Fits on 32GB GPU)
MAX_JOBS=10

# Create logs directory
mkdir -p logs_sensitivity

echo "=================================================================="
echo "Phase 1: Generating Config Files"
echo "=================================================================="

generate_yaml() {
    local dataset=$1
    local cap=$2
    local output_file=$3
    
    $PYTHON_EXEC -c "
import yaml
import os

target_file = '${output_file}'
pno_config_path = 'configs/PatchTST/${dataset}/pno.yaml'

try:
    with open(pno_config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['Experiment']['exp_id'] = f'PatchTST/${dataset}_mpc_${cap}cap'
    config['Experiment']['prev_exp_id'] = f'PatchTST/${dataset}_pno'
    config['Experiment']['mpc_first_step_cap'] = float(${cap})

    if 'Hardware' not in config: config['Hardware'] = {}
    config['Hardware']['num_workers'] = 0 
    if 'Allocate' not in config: config['Allocate'] = {}
    config['Allocate']['batch_size'] = 32
    config['Allocate']['uncertainty_quantile'] = 0.3 

    with open(target_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f'Generated: {target_file}')

except FileNotFoundError:
    print(f'[SKIP] Config not found: {pno_config_path}')
except Exception as e:
    print(f'[ERROR] Failed to generate {target_file}: {e}')
"
}

# Generate all configs
for DATASET in "${DATASETS[@]}"; do
    for CAP in "${CAPS[@]}"; do
        CONFIG_FILE="configs/PatchTST/${DATASET}/temp_mpc_${CAP}.yaml"
        generate_yaml "$DATASET" "$CAP" "$CONFIG_FILE"
    done
done

echo ""
echo "=================================================================="
echo "Phase 2: Running Batched Experiments"
echo "=================================================================="

counter=0

for DATASET in "${DATASETS[@]}"; do
    # Check if PnO model exists
    PNO_MODEL="output/PatchTST/${DATASET}_pno/model.pt"
    if [ ! -f "$PNO_MODEL" ]; then
        echo "[WARNING] PnO model not found for $DATASET! Skipping..."
        continue
    fi

    for CAP in "${CAPS[@]}"; do
        CONFIG_FILE="configs/PatchTST/${DATASET}/temp_mpc_${CAP}.yaml"
        LOG_FILE="logs_sensitivity/${DATASET}_mpc_${CAP}.log"
        
        if [ -f "$CONFIG_FILE" ]; then
            echo " > Launching: $DATASET (Cap $CAP)"
            
            # Run in background
            nohup $PYTHON_EXEC src/run_mpc.py --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
            
            # --- QUEUE LOGIC ---
            ((counter++))
            
            # If we reached MAX_JOBS, wait for them to finish before starting new ones
            if (( counter >= MAX_JOBS )); then
                echo "   [Batch Limit Reached] Waiting for current batch of $MAX_JOBS jobs to finish..."
                wait
                echo "   [Resuming] Batch finished. Starting next batch..."
                counter=0
            fi
        fi
    done
done

# Wait for the final stragglers
wait

echo ""
echo "=================================================================="
echo "Phase 3: Cleanup"
echo "=================================================================="
rm configs/PatchTST/*/temp_mpc_*.yaml
echo "Cleaned up temporary config files."
echo "Done."