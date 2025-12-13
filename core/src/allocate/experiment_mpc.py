import os
import json
import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from tqdm import tqdm

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
        # Transpose check for safety (if saved as [H, 1])
        if self.constraint.shape[0] != 1 and self.constraint.shape[1] == 1:
             self.constraint = self.constraint.T

    def _setup_gurobi_env(self):
        # Global Gurobi settings for determinism
        gp.setParam("OutputFlag", 0) # Suppress Gurobi log output
        gp.setParam("Threads", 1)    # Use a single thread
        gp.setParam("Method", 1)     # Dual Simplex (a deterministic method)
        gp.setParam("Crossover", 0)
        gp.setParam("Seed", getattr(self.configs, "random_seed", 42))
    
    def _build_allocate_model(self):
        self.allocate_model = AllocateModel(
            self.constraint.numpy(),
            self.configs.uncertainty_quantile,
            pred_len=self.configs.pred_len, # Max horizon
            first_step_cap=getattr(self.configs, "mpc_first_step_cap", None),
            quiet=True
        )

    def evaluate(self):
        print(f"Starting MPC Evaluation on {len(self.test_loader)} batches...")
        
        total_regret = 0.0
        total_rel_regret = 0.0
        count = 0
        #TODO: logger

        # Setup Logger
        cases_dir = os.path.join(self.exp_dir, "cases_test")
        os.makedirs(cases_dir, exist_ok=True)
        logger = CaseLogger(os.path.join(cases_dir, "mpc_cases.jsonl"))
        
        for batch in tqdm(self.test_loader):
            # batch_x: Normalized history [B, Seq, 1]
            # batch_y: Normalized future [B, Pred, 1]
            batch_x, batch_y, _, _ = batch
            
            batch_x_np = batch_x.cpu().numpy()
            batch_y_np = batch_y.cpu().numpy()
            
            # Unscale ground truth future prices once, as this is used in the simulation
            batch_y_real = np.zeros_like(batch_y_np)
            for i in range(len(batch_y_np)):
                batch_y_real[i] = self.scaler.inverse_transform(batch_y_np[i].reshape(-1, 1)).flatten()

            # Run simulation for each sample in batch
            for i in range(len(batch_x_np)):
                history = batch_x_np[i].flatten()
                future = batch_y_real[i].flatten()
                
                # --- CORE CALL ---
                cost, alloc = self._run_mpc_simulation(history, future)
                # -----------------
                
                # Calculate metrics
                optimal_cost = np.min(future) # Optimal buyer buys at the minimum future price
                regret = cost - optimal_cost
                rel_regret = regret / optimal_cost if optimal_cost != 0 else 0
                
                total_regret += regret
                total_rel_regret += rel_regret
                count += 1
                
                #TODO: logger
                # Log
                logger.log({
                    "case_id": count,
                    "algo": "mpc",
                    "regret": regret,
                    "rel_regret": rel_regret,
                    "alloc": alloc,
                    "true_prices": future.tolist()
                })

        avg_regret = total_regret / count
        avg_rel_regret = total_rel_regret / count

        # --- NEW: Construct standardized result.json ---
        
        # 1. Try to get MSE/MAE from the baseline (since model is identical)
        base_metrics = {}
        try:
            base_res_path = os.path.join(self.prev_exp_dir, 'result.json')
            if os.path.exists(base_res_path):
                with open(base_res_path, 'r') as f:
                    base_metrics = json.load(f)
        except Exception:
            pass

        # 2. Construct final dictionary
        final_metrics = {
            'MSE': base_metrics.get('MSE', -1.0),       # Copy from baseline
            'MAE': base_metrics.get('MAE', -1.0),       # Copy from baseline
            'Regret': avg_regret,                       # Our new MPC result
            'Rel Regret': avg_rel_regret                # Our new MPC result
        }

        print(f"Final MPC Results: {final_metrics}")
        
        # 3. Save as standard 'result.json'
        res_path = os.path.join(self.exp_dir, 'result.json')
        with open(res_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
            
        return avg_regret

    def _run_mpc_simulation(self, history_scaled, future_unscaled):
        H = self.configs.pred_len # Should be 88
        
        # State variables
        # current_history is unscaled (real prices)
        current_history_unscaled = list(self.scaler.inverse_transform(history_scaled.reshape(-1, 1)).flatten())
        budget_remaining = 1.0
        cost_incurred = 0.0
        actions_taken = []
        
        # Loop through each step in the prediction horizon
        for t in range(H):
            # 1. Prepare history for forecasting
            # Use the last 'seq_len' real prices (the sliding window)
            hist_unscaled = np.array(current_history_unscaled[-self.configs.seq_len:]).reshape(-1, 1)
            hist_norm = self.scaler.transform(hist_unscaled)
            hist_tensor = torch.tensor(hist_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 2. Forecast
            with torch.no_grad():
                pred_norm = self.forecast_model(hist_tensor) # [1, H, 1]
            
            # Inverse transform the forecast to get real price predictions
            pred_norm_np = pred_norm.cpu().numpy().flatten()
            pred_real = self.scaler.inverse_transform(pred_norm_np.reshape(-1, 1)).flatten()
            
            # 3. Optimization (The MPC Step)
            
            # A. Determine remaining horizon and constraint
            remaining_steps = H - t
            
            if t == H - 1:
                # If it's the last step, we must spend everything
                amount_to_spend_fraction = budget_remaining
                
            else:
                # B. Get the relevant constraint vector for the remaining steps
                # We use the first 'remaining_steps' from the full constraint vector r (r[0]...r[H-t])
                current_constraint = self.constraint[:, :remaining_steps].numpy()
                
                # C. Re-initialize solver for the current look-ahead window
                # We pass the first-step cap here.
                solver = AllocateModel(
                    current_constraint,
                    self.configs.uncertainty_quantile,
                    pred_len=remaining_steps,
                    first_step_cap=getattr(self.configs, "mpc_first_step_cap", None),
                    quiet=True
                )
                
                # D. Solve: Objective is the predicted prices for the remaining steps
                solver.setObj(pred_real[:remaining_steps])
                sol, _ = solver.solve() # sol is the allocation vector for the remaining steps
                
                # E. Extract the decision: We only care about the *first* action in the plan
                # sol[0] is the fraction of the TOTAL portfolio (1.0) to spend NOW.
                raw_action_fraction = float(sol[0])
                
                # F. Cap the raw action by the remaining budget
                # We must not try to spend more than we have left!
                amount_to_spend_fraction = min(raw_action_fraction, budget_remaining)
                
            # 4. Execute (Trade)
            
            # Get the true price for the current step t
            true_price = future_unscaled[t]
            
            # The actual execution cost is the *fraction of remaining budget* times the true price
            cost_incurred += amount_to_spend_fraction * true_price
            budget_remaining -= amount_to_spend_fraction
            actions_taken.append(amount_to_spend_fraction)
            
            # 5. Update History (Slide window for the next step)
            current_history_unscaled.append(true_price)
            
        return cost_incurred, actions_taken

        