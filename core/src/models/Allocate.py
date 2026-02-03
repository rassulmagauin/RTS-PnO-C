import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb import optGrbModel

class AllocateModel(optGrbModel):
    def __init__(
        self, 
        uncertainty, 
        uncertainty_quantile=0.5, 
        pred_len=None, 
        first_step_cap=None,
        seed=42,
        threads=1,
        method=1,
        crossover=0,
        quiet=False
        ):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_quantile = uncertainty_quantile
        self.uncertainty_bar = np.quantile(self.uncertainty, self.uncertainty_quantile)
        # print(self.uncertainty_bar)
        if pred_len is not None:
            self.pred_len = int(pred_len)
        else:
            self.pred_len = len(self.uncertainty[0])

        self.first_step_cap = None
        if first_step_cap is not None:
            self.first_step_cap = float(max(0.0, min(1.0, first_step_cap)))
        self.seed = seed
        self.threads = threads
        self.method = method
        self.crossover = crossover
        self.quiet = quiet
        super().__init__()

    def update_uncertainty(self, uncertainty):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_quantile = np.quantile(self.uncertainty, self.uncertainty_quantile)
        # print(self.uncertainty_bar)

    def _getModel(self): 
        # create a model
        optmodel = gp.Model()

        optmodel.Params.OutputFlag = 0 if self.quiet else 1
        optmodel.Params.Threads = self.threads
        optmodel.Params.Seed = self.seed
        optmodel.Params.Method = self.method
        optmodel.Params.Crossover = self.crossover
        # variables
        action = optmodel.addVars(self.pred_len, name="action", vtype=GRB.CONTINUOUS)
        # model sense
        optmodel.ModelSense = GRB.MINIMIZE
        # constraints
        optmodel.addConstr(gp.quicksum(self.uncertainty[0,i] * action[i] for i in range(self.pred_len)) <= self.uncertainty_bar)
        optmodel.addConstr(gp.quicksum(action[i] for i in range(self.pred_len)) == 1)
        if self.first_step_cap is not None and self.pred_len > 1:
            optmodel.addConstr(
                action[0] <= self.first_step_cap, 
                name="first_step_safety_cap"
            )
        # This handles cases where uncertainty vector might be larger than current pred_len
        # (Though with our slicing logic later, this is mostly a fallback)
        total_uncertainty_len = len(self.uncertainty[0])
        if self.pred_len < total_uncertainty_len:
             for i in range(self.pred_len, total_uncertainty_len):
                 optmodel.addConstr(action[i] == 0.0)
        return optmodel, action
    
    def solve(self):
        """
        Robust solve method that handles Infeasible models gracefully.
        Overrides the parent class method to prevent crashes on high-volatility data.
        """
        try:
            # Try the standard solver from the parent class (optGrbModel)
            return super().solve()
        except Exception:
            # FALLBACK: If Infeasible (Gurobi crash), return 0.0 allocation.
            # We return a list of zeros matching the prediction length.
            return [0.0] * self.pred_len, 0.0

class AllocateModelOld(optGrbModel):
    def __init__(self, uncertainty, uncertainty_bar):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_bar = uncertainty_bar
        self.pred_len = len(self.uncertainty[0])
        super().__init__()

    def _getModel(self): 
        # create a model
        optmodel = gp.Model()
        # variables
        action = optmodel.addVars(self.pred_len, name="action", vtype=GRB.CONTINUOUS)
        # model sense
        optmodel.ModelSense = GRB.MINIMIZE
        # constraints
        optmodel.addConstr(gp.quicksum(self.uncertainty[0,i] * action[i] for i in range(self.pred_len)) <= self.uncertainty_bar)
        optmodel.addConstr(gp.quicksum(action[i] for i in range(self.pred_len)) == 1)
        return optmodel, action
