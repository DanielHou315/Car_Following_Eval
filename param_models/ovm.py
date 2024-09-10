"""
This is the abstract class
"""
import os
from math import *
import numpy as np
import matplotlib.pyplot as plt
from .param_base_model import *


# Implemetation provided by this zotero://select/library/items/F8J5G45F
OVM_def_properties = HyperparamCollection(
    [
        Hyperparam("alpha", HP_REAL, 1/5.12, 0.1, 2.0),
        Hyperparam("v_max", HP_REAL, 36.13, 1.0, 40.0),
        Hyperparam("s_0", HP_REAL, 4, 1.0, 30.0),
        Hyperparam("theta", HP_REAL, 9.41, 1.0, 30.0),
        Hyperparam("beta", HP_REAL, 0.10, 0.001, 10.0)
    ]
)

# OVM overall calib S RMSNE [8.30666840e-01, 1.51769309e+01, 1.00000000e-02, 2.83147426, 1.22445571e+01, 3.89127702e-01]

class BatchOVM(ParamBaseModelBatch):
    def __init__(self, 
                 dfs, 
                 name="OVM", 
                 props=OVM_def_properties, 
                 pred_field=PRED_ACC, 
                 mode="gpu", 
                 exclude_ids=[], 
                 include_ids=[], 
                 flatten=False, 
                 chunk_len=None, 
                 test_ratio=0):
        super(BatchOVM, self).__init__(dfs, name, props, pred_field, mode, exclude_ids, include_ids, flatten, chunk_len, test_ratio)
        self.computed_fields = 1
    
    def v_opt(self, t, hp):
        return hp['v_max'] * (torch.tanh((self.s_pred[...,t]-hp["s_0"])/hp['theta'] - hp['beta']) + torch.tanh(hp['beta'])) / (1 + torch.tanh(hp["beta"]))
    
    def calc_a(self, t, hp):
        self.v_opt_hist[...,t] = self.v_opt(t, hp)
        return torch.clip((hp['alpha'] * (self.v_opt_hist[..., t] - self.v_pred[..., t]))
                          , self.ONE*B_MAX_PHYS, self.ONE*A_MAX_PHYS)
    
    def init_sim_matr(self, batch_size, hp):
        # Call super first, then see if more initialization is needed
        shape = super().init_sim_matr(batch_size, hp)
        if shape is None:
            return
        self.v_opt_hist = torch.ones(size=shape, dtype=torch.float32, device=self.device)
        
    def plot_computed(self, axs, s, show=False):
        """
        Plot computed values of 
        """
        assert(len(axs) >= 1)
        self.plot_t_X(axs[0], s,'t','Opt vs Actual Velocity (m/s)', self.v_opt_hist, gt=self.v_pred)
        
