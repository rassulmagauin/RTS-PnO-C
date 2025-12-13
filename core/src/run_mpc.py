import os
import random
from argparse import Namespace, ArgumentParser
import yaml
import numpy as np
import torch

# 1. Enforce Determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# 2. Import your new Experiment
from allocate.experiment_mpc import MPCExperiment

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as fin:
        configs = yaml.safe_load(fin)

    # Create output directory based on exp_id
    exp_dir = os.path.join('output', configs['Experiment']['exp_id'])
    os.makedirs(exp_dir, exist_ok=True)

    # Flatten config dictionary into a Namespace object
    configs = Namespace(**{
        arg: val
        for _, args in configs.items()
        for arg, val in args.items()
    })
    
    # 3. Run the MPC Experiment
    exp = MPCExperiment(configs)
    exp.evaluate()