"""
This is the abstract class
"""
import os
from math import floor
import numpy as np
import torch
import matplotlib.pyplot as plt

from .param_base_model import *


Gipps_def_properties = HyperparamCollection(
    [
        Hyperparam("a_opt", HP_REAL, 1.24, 0.1, 10.0),         # From zotero://select/library/items/V9IHWAFA
        Hyperparam("b_opt", HP_REAL, -2.57, -10.0, -0.1),        # Maximum braking power, from zotero://select/library/items/V9IHWAFA
        Hyperparam("tau", HP_CONST, 1.02),             # 1.02 seconds was zotero://select/library/items/V9IHWAFA
        Hyperparam("v_opt", HP_REAL, 41.88, 10.0, 50.0),         # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("s_0", HP_REAL, 7.83, 1.0, 50.0),
        Hyperparam("lv_b_opt", HP_REAL, -2, -10.0, -0.5),
        # We don't need a 6-th vale, we can compute b_hat, rather than using a fixed value,
    ]
)

class BatchGipps(ParamBaseModelBatch):
    def __init__(self,
                 dataset, 
                 name="Gipps", 
                 props=Gipps_def_properties, 
                 pred_field=PRED_VEL,
                 mode="gpu", 
                 exclude_ids=[], 
                 include_ids=None, 
                 flatten=False, 
                 chunk_len=None, 
                 test_ratio=0):
        super(BatchGipps, self).__init__(dataset, name, props, pred_field, mode, exclude_ids, include_ids, flatten, chunk_len, test_ratio)
        self.computed_fields = 4
    
    def calc_v_acc(self, t, hp):
        """Compute accelerate portion of Gipps"""
        # Should be good to go
        t_prev = t - self.tau_ticks
        v_frac = torch.clip((self.v_pred[...,t_prev] / hp['v_opt']), self.ONE*-128, self.ONE*128)
        p2 =  2.5*hp['a_opt']*hp['tau']*(1-v_frac)*torch.sqrt(0.025 + v_frac)
        self.v_acc[..., t] = self.v_pred[..., t_prev] + p2
        return self.v_acc[...,t]
    
    def calc_v_dec(self, t, hp):
        # Compute Chunk
        t_prev = t - self.tau_ticks
        chunk = (hp['b_opt']*hp['tau'])**2 - hp['b_opt'] \
                * (2 * (self.s_pred[..., t_prev] - hp['s_0'])
                   - self.v_pred[..., t_prev] * hp['tau'] 
                   - torch.clip((self.lv_v[..., t_prev]**2 / hp["lv_b_opt"]), self.ONE * -8, self.ONE * 8)
                  )
        assert(torch.isnan(chunk).any() == False)
        # Make sure what gets computed make sense
        self.v_dec[...,t] = hp['b_opt'] * hp['tau'] + torch.sqrt(torch.maximum(self.ZERO, chunk))
        return self.v_dec[...,t]
    
    
    def calc_v(self, t, hp):
        return torch.clip(torch.minimum(self.calc_v_acc(t+1,hp), self.calc_v_dec(t+1, hp)), self.ZERO, self.ONE*100)
    
    def init_sim_matr(self,batch_size, hp):
        # Call super first, then see if more initialization is needed
        shape = super().init_sim_matr(batch_size, hp)
        if shape is None:
            return
        self.v_acc = torch.ones(size=shape, dtype=torch.float32, device=self.device)
        self.v_dec = torch.ones(size=shape, dtype=torch.float32, device=self.device)
        
        self.tau_ticks = floor(hp['tau'].detach().flatten()[0].item() * 10)
        self.start_sim_time = self.tau_ticks
        self.end_sim_time = self.end_sim_time - 1
        
        self.a_pred[..., :self.tau_ticks] = self.a[..., :self.tau_ticks]
        self.v_pred[..., :self.tau_ticks] = self.v[..., :self.tau_ticks]
        self.s_pred[..., :self.tau_ticks] = self.s[..., :self.tau_ticks]
        self.x_pred[..., :self.tau_ticks] = self.x[..., :self.tau_ticks]
        self.relv_pred[..., :self.tau_ticks] = self.relv[..., :self.tau_ticks]
        
        # self.v_frac = torch.ones(size=shape, dtype=torch.float32, device=self.device)
        # self.p2 = torch.ones(size=shape, dtype=torch.float32, device=self.device)
        
        
    # def simulate_step(self, t, hp):
    #     self.v_pred[..., t] = torch.clip(torch.minimum(self.calc_v_acc(t,hp), self.calc_v_dec(t, hp)), self.ZERO, self.ONE*100)
    #     self.a_pred[..., t] = self.TICK * self.tau_ticks * (self.v_pred[..., t] - self.v_pred[..., t-self.tau_ticks])   # Average acceleration
    #     self.x_pred[..., t] = self.x_pred[..., t-1] + self.v_pred[..., t] * self.TICK + self.a_pred[..., t] * self.TICK**2 / 2.0
    #     self.s_pred[..., t] = self.lv_x[..., t] - self.x_pred[..., t]
    #     self.relv_pred[..., t+1] = self.lv_v[..., t] - self.v_pred[..., t]
                
    def plot_computed(self, axs, s, show=False):
        """
        Plot computed values of 
        """
        assert(len(axs) >= 4)
        self.plot_t_X(axs[0], s,'t','V Acceleration', self.v_acc)
        self.plot_t_X(axs[1], s,'t','V Deceleration', self.v_dec)
        # self.plot_t_X(axs[2], s,'t','V Fraction', self.v_frac)
        # self.plot_t_X(axs[3], s,'t','V Component', self.p2)
