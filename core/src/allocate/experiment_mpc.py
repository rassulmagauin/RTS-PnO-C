import os
import json
import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from tqdm import tqdm
from joblib import Parallel, delayed

import models
from models.Allocate import AllocateModel
from common.experiment import Experiment
from allocate.data_provider import get_allocating_loader_and_dataset
from utils.case_logger import CaseLogger

class MPCExperiment(Experiment):
    def __init__(self, configs):
        super().__init__(configs)
        self.prev_exp_dir = os.path.join('output', configs.prev_exp_id)
        self._build_forecast_model()
        self._load_constraint()
        self._setup_gurobi_env()
        self._build_allocate_model()

        self.test_loader, self.test_set = get_allocating_loader_and_dataset(
            self.configs, self.allocate_model, split='test', shuffle=False
        )
        self.scaler = self.test_set.scaler

    def _build_forecast_model(self):
        self.forecast_model = getattr(models, self.configs.model)(self.configs)
        if self.configs.use_multi_gpu:
            self.forecast_model = nn.DataParallel(self.forecast_model, device_ids=self.configs.gpus)
        self.forecast_model.to(self.device)
        model_path = os.path.join(self.prev_exp_dir, 'model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
        print(f"Loading pre-trained model from: {model_path}")
        self.forecast_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.forecast_model.eval()
        self.model = self.forecast_model

    def _load_constraint(self):
        constraint_path = os.path.join(self.prev_exp_dir, 'constraint.pt')
        if not os.path.exists(constraint_path):
             constraint_path = os.path.join(self.prev_exp_dir, 'score.pt')
        if not os.path.exists(constraint_path):
             raise FileNotFoundError(f"Constraint file not found at {constraint_path}")
        print(f"Loading constraints from: {constraint_path}")
        self.constraint = torch.load(constraint_path, map_location='cpu')
        if self.constraint.dim() == 1:
             self.constraint = self.constraint.unsqueeze(0)
        if self.constraint.shape[0] != 1 and self.constraint.shape[1] == 1:
             self.constraint = self.constraint.T

    def _setup_gurobi_env(self):
        # Global Gurobi settings for determinism
        gp.setParam("OutputFlag", 0) 
        gp.setParam("Threads", 1)    
        gp.setParam("Method", 1)     
        gp.setParam("Crossover", 0)
        gp.setParam("Seed", getattr(self.configs, "random_seed", 42))
    
    def _build_allocate_model(self):
        # Template initialization
        self.allocate_model = AllocateModel(
            self.constraint.numpy(),
            self.configs.uncertainty_quantile,
            pred_len=self.configs.pred_len,
            first_step_cap=getattr(self.configs, "mpc_first_step_cap", None),
            quiet=True
        )

    def evaluate(self):
        print(f"Starting MPC Evaluation on {len(self.test_loader)} batches...")
        
        total_regret = 0.0
        total_rel_regret = 0.0
        count = 0

        cases_dir = os.path.join(self.exp_dir, "cases_test")
        os.makedirs(cases_dir, exist_ok=True)
        logger = CaseLogger(os.path.join(cases_dir, "mpc_cases.jsonl"))
        
        print("Parallelizing execution across available cores (backend='threading')...")
        
        for batch in tqdm(self.test_loader):
            batch_x, batch_y, _, _ = batch
            
            batch_x_np = batch_x.cpu().numpy()
            batch_y_np = batch_y.cpu().numpy()
            
            batch_y_real = np.zeros_like(batch_y_np)
            for i in range(len(batch_y_np)):
                batch_y_real[i] = self.scaler.inverse_transform(batch_y_np[i].reshape(-1, 1)).flatten()

            # --- PARALLEL LOOP START ---
            # Using 'threading' backend is safer for PyTorch shared memory
            results = Parallel(n_jobs=4, backend="threading")(
                delayed(self._run_mpc_simulation)(
                    batch_x_np[i].flatten(), 
                    batch_y_real[i].flatten()
                ) for i in range(len(batch_x_np))
            )
            # --- PARALLEL LOOP END ---

            for i, (cost, alloc, preds) in enumerate(results):
                future = batch_y_real[i].flatten()
                
                optimal_cost = np.min(future)
                regret = cost - optimal_cost
                rel_regret = regret / optimal_cost if optimal_cost != 0 else 0
                
                total_regret += regret
                total_rel_regret += rel_regret
                count += 1
                
                logger.log({
                    "case_id": count,
                    "algo": "mpc",
                    "regret": regret,
                    "rel_regret": rel_regret,
                    "alloc": alloc,
                    "true_prices": future.tolist(),
                    "pred_prices": preds
                })

        avg_regret = total_regret / count
        avg_rel_regret = total_rel_regret / count

        base_metrics = {}
        try:
            base_res_path = os.path.join(self.prev_exp_dir, 'result.json')
            if os.path.exists(base_res_path):
                with open(base_res_path, 'r') as f:
                    base_metrics = json.load(f)
        except Exception:
            pass

        final_metrics = {
            'MSE': base_metrics.get('MSE', -1.0),
            'MAE': base_metrics.get('MAE', -1.0),
            'Regret': avg_regret,
            'Rel Regret': avg_rel_regret
        }

        print(f"Final MPC Results: {final_metrics}")
        
        res_path = os.path.join(self.exp_dir, 'result.json')
        with open(res_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
            
        return avg_regret

    def _run_mpc_simulation(self, history_norm_flat, future_real_flat):
        H = self.configs.pred_len

        liquidation_window = max(5, int(H * 0.1)) 
        liquidation_start_step = H - liquidation_window
        
        current_history_unscaled = list(self.scaler.inverse_transform(history_norm_flat.reshape(-1, 1)).flatten())
        budget_remaining = 1.0
        cost_incurred = 0.0
        actions_taken = []

        rolling_preds = []
        
        for t in range(H):
            hist_unscaled = np.array(current_history_unscaled[-self.configs.seq_len:]).reshape(-1, 1)
            hist_norm = self.scaler.transform(hist_unscaled)
            hist_tensor = torch.tensor(hist_norm, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                hist_tensor = hist_tensor.to(self.device)
                pred_norm = self.model(hist_tensor)
            
            pred_norm_np = pred_norm.cpu().numpy().flatten()
            pred_real = self.scaler.inverse_transform(pred_norm_np.reshape(-1, 1)).flatten()
            
            rolling_preds.append(float(pred_real[0]))

            remaining_steps = H - t
            
            if t == H - 1:
                amount_to_spend_fraction = budget_remaining
            else:
                current_constraint = self.constraint[:, :remaining_steps].numpy()

                current_cap = getattr(self.configs, "mpc_first_step_cap", None)
                if t >= liquidation_start_step:
                    current_cap = None
                raw_action_fraction = 0.0
                
                try:
                    solver = AllocateModel(
                        current_constraint,
                        self.configs.uncertainty_quantile,
                        pred_len=remaining_steps,
                        first_step_cap=current_cap,
                        quiet=True
                    )
                    solver.setObj(pred_real[:remaining_steps])
                    sol, _ = solver.solve()
                    raw_action_fraction = float(sol[0])
                except Exception:
                    try:
                        solver_relaxed = AllocateModel(
                            current_constraint,
                            self.configs.uncertainty_quantile,
                            pred_len=remaining_steps,
                            first_step_cap=None,
                            quiet=True
                        )
                        solver_relaxed.setObj(pred_real[:remaining_steps])
                        sol, _ = solver_relaxed.solve()
                        raw_action_fraction = float(sol[0])
                    except Exception:
                        raw_action_fraction = 0.0
                if t >= liquidation_start_step and raw_action_fraction < 1e-6:
                    
                    # 1. Look at the forecast for the remaining liquidation period
                    future_preds = pred_real[:remaining_steps]
                    
                    # 2. Compare "Now" vs "Future Average"
                    current_pred_price = future_preds[0]
                    avg_future_price = np.mean(future_preds)
                    
                    # 3. Calculate Urgency
                    # > 1.0 : Prices are rising. Buy MORE now.
                    # < 1.0 : Prices are falling. Buy LESS now (wait).
                    if current_pred_price > 1e-6: # Avoid division by zero
                        urgency_factor = avg_future_price / current_pred_price
                    else:
                        urgency_factor = 1.0
                        
                    # 4. Clip the factor for safety 
                    # We never want to stop completely (0.5x) or panic dump (2.0x)
                    urgency_factor = np.clip(urgency_factor, 0.5, 2.0)
                    
                    # 5. Apply to the Base Pace (TWAP)
                    base_pace = budget_remaining / remaining_steps
                    raw_action_fraction = base_pace * urgency_factor
                amount_to_spend_fraction = min(raw_action_fraction, budget_remaining)
            
            true_price = future_real_flat[t]
            cost_incurred += amount_to_spend_fraction * true_price
            budget_remaining -= amount_to_spend_fraction
            actions_taken.append(amount_to_spend_fraction)
            
            current_history_unscaled.append(true_price)
            
        return cost_incurred, actions_taken, rolling_preds