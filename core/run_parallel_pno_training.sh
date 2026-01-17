#!/bin/bash

# ==============================================================================
# PARALLEL PnO TRAINING (Constraint-Aware)
# ==============================================================================

# 1. Define Datasets and Caps
DATASETS=("usdcny" "coinbase" "audusd" "djia" "sp500" "usdjpy")
CAPS=(0.1 0.25 0.5 0.75 1.0)
PYTHON_EXEC="python"

# --- MAX PARALLEL JOBS ---
# 10 Jobs * 4 Threads = 40 Cores used (Fits 48-core machine).
MAX_JOBS=10

# Create directories
mkdir -p logs_training
mkdir -p configs/temp_pno_training

echo "=================================================================="
echo "Phase 1: Generating Training Configs"
echo "=================================================================="

generate_train_config() {
    local dataset=$1
    local cap=$2
    local output_file=$3
    
    $PYTHON_EXEC -c "
import yaml
import os

target_file = '${output_file}'
base_config = 'configs/PatchTST/${dataset}/pno.yaml'

try:
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Set Unique Experiment ID
    config['Experiment']['exp_id'] = f'PatchTST/${dataset}_pno_${cap}cap'
    
    # 2. Inject the Constraint Cap into Training
    config['Experiment']['mpc_first_step_cap'] = float(${cap})

    # 3. Optimize Hardware for Parallelism
    if 'Hardware' not in config: config['Hardware'] = {}
    config['Hardware']['num_workers'] = 0  # Crucial: prevents DataLoader forks
    
    # --- CHANGE: REMOVED EPOCH LIMITS ---
    # We want full training to match the baseline.
    # config['Allocate']['train_epochs'] = 20 
    # config['Allocate']['patience'] = 3

    with open(target_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f'Generated: {target_file}')

except FileNotFoundError:
    print(f'[SKIP] Config not found: {base_config}')
except Exception as e:
    print(f'[ERROR] Failed to generate {target_file}: {e}')
"
}

# Generate all configs upfront
for DATASET in "${DATASETS[@]}"; do
    for CAP in "${CAPS[@]}"; do
        CONFIG_FILE="configs/temp_pno_training/${DATASET}_train_${CAP}.yaml"
        generate_train_config "$DATASET" "$CAP" "$CONFIG_FILE"
    done
done

echo ""
echo "=================================================================="
echo "Phase 2: Launching Parallel Training"
echo "=================================================================="

counter=0

# Loop through datasets and caps to launch jobs
for DATASET in "${DATASETS[@]}"; do
    for CAP in "${CAPS[@]}"; do
        CONFIG_FILE="configs/temp_pno_training/${DATASET}_train_${CAP}.yaml"
        LOG_FILE="logs_training/pno_train_${DATASET}_${CAP}.log"
        
        if [ -f "$CONFIG_FILE" ]; then
            echo " > Training: $DATASET (Cap $CAP) -> Logs: $LOG_FILE"
            
            # --- CRITICAL: Force PyTorch to use only 4 threads per job ---
            export OMP_NUM_THREADS=4 
            
            nohup $PYTHON_EXEC src/run_pno.py --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
            
            ((counter++))
            
            # Batch Limiter
            if (( counter >= MAX_JOBS )); then
                echo "   [Limit Reached] Waiting for batch of $MAX_JOBS models to finish..."
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
rm configs/temp_pno_training/*.yaml
echo "Done. All constraint-aware PnO models are trained."