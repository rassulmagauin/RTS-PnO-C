#!/bin/bash

# ==============================================================================
# PARALLEL MPC SENSITIVITY EVALUATION
# ==============================================================================

# 1. Define Datasets and Caps
DATASETS=("coinbase" "audusd" "djia" "sp500" "usdjpy")
CAPS=(0.1 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.9 1.0)
PYTHON_EXEC="python"

# --- MAX PARALLEL JOBS ---
MAX_JOBS=10

# Create directories
mkdir -p logs_mpc
mkdir -p configs/temp_mpc_sensitivity

echo "=================================================================="
echo "Phase 1: Generating MPC Configs"
echo "=================================================================="

generate_mpc_config() {
    local dataset=$1
    local cap=$2
    local output_file=$3
    
    $PYTHON_EXEC -c "
import yaml
import os

target_file = '${output_file}'

# --- FIX: Fallback Logic ---
# Try to find mpc.yaml. If missing, use pno.yaml as the template.
base_mpc = 'configs/PatchTST/${dataset}/mpc.yaml'
base_pno = 'configs/PatchTST/${dataset}/pno.yaml'

if os.path.exists(base_mpc):
    base_config = base_mpc
elif os.path.exists(base_pno):
    base_config = base_pno
else:
    raise FileNotFoundError(f'Neither mpc.yaml nor pno.yaml found in configs/PatchTST/${dataset}/')

try:
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Point to the SPECIFIC trained model we just finished
    # e.g. output/PatchTST/usdcny_pno_0.1cap
    config['Experiment']['prev_exp_id'] = f'PatchTST/${dataset}_pno_${cap}cap'

    # 2. Set a new unique ID for this MPC test result
    config['Experiment']['exp_id'] = f'PatchTST/${dataset}_mpc_${cap}cap'
    
    # 3. Enforce the same Constraint Cap during Testing
    config['Experiment']['mpc_first_step_cap'] = float(${cap})

    # 4. Hardware Optimization
    if 'Hardware' not in config: config['Hardware'] = {}
    config['Hardware']['num_workers'] = 0 
    
    # 5. Ensure we load weights
    if 'Allocate' not in config: config['Allocate'] = {}
    config['Allocate']['load_prev_weights'] = True

    with open(target_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f'Generated: {target_file}')

except Exception as e:
    print(f'[ERROR] Failed to generate {target_file}: {e}')
"
}

# Generate all configs upfront
for DATASET in "${DATASETS[@]}"; do
    for CAP in "${CAPS[@]}"; do
        CONFIG_FILE="configs/temp_mpc_sensitivity/${DATASET}_mpc_${CAP}.yaml"
        generate_mpc_config "$DATASET" "$CAP" "$CONFIG_FILE"
    done
done

echo ""
echo "=================================================================="
echo "Phase 2: Launching Parallel MPC Evaluation"
echo "=================================================================="

counter=0

# Loop through datasets and caps to launch jobs
for DATASET in "${DATASETS[@]}"; do
    for CAP in "${CAPS[@]}"; do
        CONFIG_FILE="configs/temp_mpc_sensitivity/${DATASET}_mpc_${CAP}.yaml"
        LOG_FILE="logs_mpc/mpc_eval_${DATASET}_${CAP}.log"
        
        if [ -f "$CONFIG_FILE" ]; then
            echo " > Testing: $DATASET (Cap $CAP) -> Logs: $LOG_FILE"
            
            export OMP_NUM_THREADS=4 
            
            nohup $PYTHON_EXEC src/run_mpc.py --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
            
            ((counter++))
            
            # Batch Limiter
            if (( counter >= MAX_JOBS )); then
                echo "   [Limit Reached] Waiting for batch of $MAX_JOBS evaluations to finish..."
                wait
                echo "   [Resuming] Batch finished. Starting next batch..."
                counter=0
            fi
        fi
    done
done

# Wait for the final batch
wait

echo ""
echo "=================================================================="
echo "Phase 3: Cleanup"
echo "=================================================================="
rm configs/temp_mpc_sensitivity/*.yaml
echo "Done. All MPC sensitivity evaluations are complete."