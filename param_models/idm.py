"""
This is the abstract class
"""
from math import *
# import numpy as np
import torch
import matplotlib.pyplot as plt
from .param_base_model import *


IDM_def_properties = HyperparamCollection(
    [
        Hyperparam("a_max", HP_REAL, 1.32, 0.5,5.0),   # Approximated from 2018 Honda Accord 0-60mph 7 second, max 75% pedal
                                                    # 1.5 was used in zotero://select/library/items/V9IHWAFA
        Hyperparam("b_comf", HP_REAL, 2.18, 0.5, 5.0),      # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("s_0", HP_REAL, 3.89, 1.0, 6.0),         # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("T", HP_REAL, 0.97, 0.1, 3.0),           # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("v_opt", HP_REAL, 22.27, 10.0, 40.0)       # According to zotero://select/library/items/V9IHWAFA, equivalent to 72kph
    ]
)

IDM_rmsne_properties = HyperparamCollection(
    [
        Hyperparam("a_max", HP_REAL, 1.06452546, 0.5,5.0),   # Approximated from 2018 Honda Accord 0-60mph 7 second, max 75% pedal
                                                    # 1.5 was used in zotero://select/library/items/V9IHWAFA
        Hyperparam("b_comf", HP_REAL, 0.5, 0.5, 5.0),      # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("s_0", HP_REAL, 4.29951906, 1.0, 6.0),         # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("T", HP_REAL, 2.05361199, 0.1, 3.0),           # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("v_opt", HP_REAL, 40, 10.0, 40.0)       # According to zotero://select/library/items/V9IHWAFA, equivalent to 72kph
    ]
)

IDM_gmm1 = HyperparamCollection(
    [
        Hyperparam("a_max", HP_CONST, 0.5),   # Approximated from 2018 Honda Accord 0-60mph 7 second, max 75% pedal
                                                    # 1.5 was used in zotero://select/library/items/V9IHWAFA
        Hyperparam("b_comf", HP_CONST, 1.61086221),      # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("s_0", HP_CONST, 3.03759098),         # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("T", HP_CONST, 0.71071213),           # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("v_opt", HP_CONST, 10)       # According to zotero://select/library/items/V9IHWAFA, equivalent to 72kph
    ]
)

IDM_gmm2 = HyperparamCollection(
    [
        Hyperparam("a_max", HP_CONST, 0.5),   # Approximated from 2018 Honda Accord 0-60mph 7 second, max 75% pedal
                                                    # 1.5 was used in zotero://select/library/items/V9IHWAFA
        Hyperparam("b_comf", HP_CONST, 1.40850304),      # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("s_0", HP_CONST, 2.77157784),         # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("T", HP_CONST, 0.74656671),           # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("v_opt", HP_CONST, 10)       # According to zotero://select/library/items/V9IHWAFA, equivalent to 72kph
    ]
)

IDM_gmm3 = HyperparamCollection(
    [
        Hyperparam("a_max", HP_CONST, 0.5),   # Approximated from 2018 Honda Accord 0-60mph 7 second, max 75% pedal
                                                    # 1.5 was used in zotero://select/library/items/V9IHWAFA
        Hyperparam("b_comf", HP_CONST, 0.5),      # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("s_0", HP_CONST, 5.26974699),         # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("T", HP_CONST, 0.1),           # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("v_opt", HP_CONST, 38.03118429)       # According to zotero://select/library/items/V9IHWAFA, equivalent to 72kph
    ]
)

IDM_gmm4 = HyperparamCollection(
    [
        Hyperparam("a_max", HP_CONST, 0.5),   # Approximated from 2018 Honda Accord 0-60mph 7 second, max 75% pedal
                                                    # 1.5 was used in zotero://select/library/items/V9IHWAFA
        Hyperparam("b_comf", HP_CONST, 3.20962184),      # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("s_0", HP_CONST, 5.83183497),         # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("T", HP_CONST, 0.1),           # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("v_opt", HP_CONST, 10)       # According to zotero://select/library/items/V9IHWAFA, equivalent to 72kph
    ]
)

# Mod 1: [ 0.5         1.61086221  3.03759098  0.71071213 10.        ]
# Mod 2: [ 0.5         1.40850304  2.77157784  0.74656671 10.        ]
# Mod 3: [ 0.5         0.5         5.26974699  0.1        38.03118429]
# Mod 4: [ 0.5         3.20962184  5.83183497  0.1        10.        ]

class BatchIDM(ParamBaseModelBatch):
    def __init__(self, 
                 dfs, 
                 name="IDM", 
                 props=IDM_def_properties, 
                 pred_field=PRED_ACC,
                 mode="gpu", 
                 exclude_ids=[], 
                 include_ids=None,
                 flatten=False,
                 chunk_len=None, 
                 test_ratio=0):
        super(BatchIDM, self).__init__(dfs, name, props, pred_field, mode, exclude_ids, include_ids, flatten, chunk_len, test_ratio)
        self.computed_fields = 4
        
    def calc_v_opt(self, t, hp):
        return hp['v_opt']

    def calc_s_opt(self, t, hp):
        return hp['s_0'] +  torch.max(self.ZERO, self.v_pred[...,t] * hp['T'] 
                        - self.v_pred[..., t] * self.relv_pred[..., t] / self.AB_CONST)
        
    def calc_a(self, t, hp):
        """simulate acceleration by model"""
        self.v_factor[...,t] = torch.max(self.ZERO, self.v_pred[..., t] / self.calc_v_opt(t, hp))**2
        self.s_opt[...,t] = self.calc_s_opt(t, hp)
        self.s_factor[...,t] = (self.s_opt[...,t] / self.s_pred[..., t]) ** 2
        return torch.clip(hp['a_max'] * (1 - self.v_factor[...,t] - self.s_factor[...,t])
                         , self.ONE*B_MAX_PHYS, self.ONE*A_MAX_PHYS)
    
    def init_sim_matr(self, batch_size, hp):
        # Call super first, then see if more initialization is needed
        shape = super().init_sim_matr(batch_size, hp)
        if shape is None:
            return
        
        # Record intermediary results
        self.s_opt = torch.ones(size=shape, dtype=torch.float32, device=self.device)
        self.v_factor = torch.ones(size=shape, dtype=torch.float32, device=self.device)
        self.s_factor = torch.ones(size=shape, dtype=torch.float32, device=self.device)

        self.AB_CONST =  self.ONE * torch.sqrt(2 * hp['a_max'] * hp['b_comf'])


#     def simulate(self, hp, batch_size=1, pbar=False):
#         # Initialize, this is copied for calibration purpose
#         super().simulate(hp, batch_size)
        
#         rg = range(self.sim_start_time, self.sim_end_time)
#         if pbar == True:
#             rg = tqdm(rg)

#         with torch.no_grad():
#             # Simulate for each step
#             for t in rg:
#                 self.a_pred[..., t] = self.calc_a(t, hp)
#                 self.v_pred[..., t+1] = torch.maximum(self.ZERO, self.TICK * self.a_pred[..., t] + self.v_pred[..., t])
#                 self.x_pred[..., t+1] = self.x_pred[..., t] + self.v_pred[..., t] * self.TICK + self.a_pred[..., t] * self.TICK**2 / 2.0
#                 self.s_pred[..., t+1] = self.lv_x[..., t+1] - self.x_pred[..., t+1]
#                 self.relv_pred[..., t+1] = self.lv_v[..., t+1] - self.v_pred[..., t+1]
#                 self.a_pred[..., self.mat_shape[-1]-1] = self.a_pred[..., self.mat_shape[-1]-2]


    def plot_computed(self, axs, s, show=False):
        """
        Plot computed values of 
        """
        assert(len(axs) >= 4)
        multiplier = (1 - self.v_factor - self.s_factor)
        
        self.plot_t_X(axs[0], s,'t','Optimal Distance (m)', self.s_opt, gt=self.s)
        self.plot_t_X(axs[1], s,'t','Velocity factor (**2)', self.v_factor)
        self.plot_t_X(axs[2], s,'t','Space factor (**2)', self.s_factor)
        self.plot_t_X(axs[3], s,'t','Acceleration factor', multiplier)