"""
This is the abstract class
"""
import os
from math import *
# import numpy as np
import torch
import matplotlib.pyplot as plt
from .param_base_model import *


# ACC as called in original (or Enhanced IDM): zotero://select/library/items/4WH7BDJ7
# IIDM with CAH as described in zotero://select/library/items/JEFKDLVK

ACC_def_properties = HyperparamCollection(
    [
        Hyperparam("a_max", HP_REAL, 1.32, 0.1,10.0),         # Approximated from 2018 Honda Accord 0-60mph 7 second, max 75% pedal
                            # 1.5 was used in zotero://select/library/items/V9IHWAFA
        Hyperparam("b_comf", HP_REAL, 2.18, 0.1, 10.0),      # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("s_0", HP_REAL, 3.89, 1.0, 10.0),         # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("T", HP_REAL, 0.97, 0.1, 5.0),           # According to zotero://select/library/items/V9IHWAFA
        Hyperparam("v_opt", HP_REAL, 22.27, 5.0, 40.0),       # According to zotero://select/library/items/V9IHWAFA, equivalent to 72kph
        Hyperparam("delta", HP_CONST, 4),
        Hyperparam('c', HP_CONST, 0.99)             # According to IIDM with CAH eval papper
    ]
)


class BatchACC(ParamBaseModelBatch):
    def __init__(self, 
                 dataset, 
                 name="ACC", 
                 props=ACC_def_properties, 
                 pred_field=PRED_ACC, 
                 mode="gpu", 
                 exclude_ids=[], 
                 include_ids=None, 
                 flatten=False, 
                 chunk_len=None, 
                 test_ratio=0):
        super(BatchACC, self).__init__(dataset, name, props, pred_field, mode, exclude_ids, include_ids, flatten, chunk_len, test_ratio)
        self.computed_fields = 3+7

    def calc_s_opt(self, t, hp):
        return hp['s_0'] + torch.maximum(self.ZERO, self.v_pred[...,t] * hp['T'] \
                - (self.v_pred[..., t] * self.relv_pred[..., t] / self.AB_CONST))
    
    def calc_a_free(self, t, hp):
        # Save divide
        v_v0_quotient = torch.clip((self.v_pred[...,t] / hp['v_opt']), self.ZERO, self.ONE*8.0)
        a_less = hp['a_max'] * (1 - v_v0_quotient**hp['delta'])
        if (torch.isnan(a_less).any() == True) or (torch.isinf(a_less).any() == True):
            print("ACC has NaN or Inf a_less at", t, a_less, "with", v_v0_quotient, hp['delta'])
            assert(False)
        
        # HEAVY nan checks because zero to the negative power is weird on CUDA
        a_more = hp['b_comf'] * (1 - torch.nan_to_num((v_v0_quotient**self.AF_PWR) * (v_v0_quotient != 0.0), 0)) 
        if (torch.isnan(a_more).any() == True) or (torch.isinf(a_more).any() == True):
            print("ACC has NaN or Inf a_more at", t, a_more)
            assert(False)
        
        is_less = (self.v_pred[...,t] < hp['v_opt'])
        # return torch.clip((a_less*is_less + a_more*(~is_less)), self.ONE*B_MAX_PHYS, self.ONE*B_MAX_PHYS)
        return (a_less*is_less + a_more*(~is_less))
    
    def calc_z(self, t, hp):
        self.s_opt[...,t] = self.calc_s_opt(t, hp)
        # Ratio is clipped to 10, representing problematic ratio computation
        return torch.clip(torch.nan_to_num(self.s_opt[...,t]/self.s_pred[...,t], 1), self.ONE*(-8), self.ONE*8)
    
    def calc_a_iidm(self, t, hp):
        z = self.calc_z(t, hp)
        a_free = self.calc_a_free(t, hp)
        
        # Compute first 2 regimes, where v < v_opt
        iidm_reg_1 = hp['a_max'] * (1 - z**2)
        pwr_2 = 2*torch.clip(torch.nan_to_num(hp['a_max'] / a_free, nan=4),self.ZERO, self.ONE*4).to(torch.int)
        
        iidm_reg_2 = a_free * (1-z**pwr_2)
        reg_12_criterion = (self.v_pred[...,t] <= hp['v_opt'])
        
        # Compute second 2 regimes, where v > v_opt
        iidm_reg_3 = a_free + (1-z**2)
        iidm_reg_4 = a_free
        reg_34_criterion = ~reg_12_criterion
        
        # Compute z-related criterion
        reg_13_criterion = (z >= 1.0)
        reg_24_criterion = ~reg_13_criterion
        
        r1 = iidm_reg_1 * reg_12_criterion * reg_13_criterion
        r2 = iidm_reg_2 * reg_12_criterion * reg_24_criterion
        r3 = iidm_reg_3 * reg_34_criterion * reg_13_criterion
        r4 = iidm_reg_4 * reg_34_criterion * reg_24_criterion
        
        # for i, r in enumerate([z, a_free, reg_1, pwr_2, reg_2, reg_12_criterion, reg_2, reg_4, reg_13_criterion, r1, r2, r3, r4]):
        #     if (torch.isnan(r).any() == True) or (torch.isinf(r).any() == True):
        #         print(f"ACC has NaN or Inf {i} at", t, r)
        #         assert(False)
        
        masked_sum = torch.clip((r1 + r2 + r3 + r4), self.ONE*B_MAX_PHYS, self.ONE*A_MAX_PHYS)
        # Clip IIDM output
        # return torch.clip((r1 + r2 + r3 + r4), self.ONE*B_MAX_PHYS, self.ONE*B_MAX_PHYS)
        return masked_sum
    

    def calc_a_cah(self, t, hp):
        z = self.calc_z(t, hp)
        a_eff_l = torch.minimum(self.lv_a[...,t], hp['a_max'])
        
        # Regime 1
        v2_a = self.v_pred[...,t]**2 * a_eff_l
        v2_2sa = self.v_pred[...,t]**2 - (2*self.s_pred[...,t]*a_eff_l)
        cah_reg_1  = torch.clip(torch.nan_to_num(v2_a/v2_2sa,0), self.ONE*B_MAX_PHYS, self.ONE*A_MAX_PHYS)

        # Regmie 2
        dv_term = (self.relv_pred[...,t])**2 * (self.relv_pred[...,t] >= 0)
        cah_reg_2  = a_eff_l - torch.clip((dv_term / (2 * self.s_pred[...,t])), self.ONE*B_MAX_PHYS, self.ONE*A_MAX_PHYS)
        
        # Combine results
        criterion = ((self.lv_v[...,t] * self.relv_pred[...,t]) <= -2*self.s_pred[...,t]*a_eff_l)
        
        # Clip CAH output
        # return torch.clip((reg_1*criterion.to(torch.int) + reg_2*(~criterion)), self.ONE*B_MAX_PHYS, self.ONE*B_MAX_PHYS)
        return (cah_reg_1 *criterion.to(torch.int) + cah_reg_2 *(~criterion))

    def calc_a(self, t, hp):
        """simulate acceleration by model"""
        self.a_iidm[...,t] = self.calc_a_iidm(t, hp)          
        self.a_cah[...,t] = self.calc_a_cah(t, hp)
        choose_iidm = (self.a_iidm[...,t] > self.a_cah[...,t])
                            
        alt_idm_pt = (self.ONE-hp['c']) * self.a_iidm[...,t]
        diff_pt = torch.clip(((self.a_iidm[...,t] - self.a_cah[...,t]) / hp['b_comf']), self.ONE*(-8), self.ONE*8.0)
        alt_cah_pt = hp['c'] * (self.a_cah[...,t] + hp['b_comf'] * torch.tanh(diff_pt))
        
        a_final = (choose_iidm * self.a_iidm[...,t] 
                + (~choose_iidm) * (alt_idm_pt+alt_cah_pt))
        
        # Clip final output
        return torch.clip(torch.nan_to_num(a_final, nan=0), self.ONE*B_MAX_PHYS, self.ONE*A_MAX_PHYS)
        # return (choose_iidm * self.a_iidm[...,t] + (~choose_iidm) * (alt_idm_pt+alt_cah_pt))

    def init_sim_matr(self, batch_size, hp):
        # Call super first, then see if more initialization is needed
        shape = super().init_sim_matr(batch_size, hp)
        if shape is None:
            return
        self.s_opt = torch.ones(size=shape, dtype=torch.float, device=self.device)
        self.a_iidm = torch.ones(size=shape, dtype=torch.float, device=self.device)
        self.a_cah = torch.ones(size=shape, dtype=torch.float, device=self.device)
        
        # Very temporary debugging things
        # self.iidm_reg_1 = torch.ones(size=shape, dtype=torch.float, device=self.device)
        # self.iidm_reg_2 = torch.ones(size=shape, dtype=torch.float, device=self.device)
        # self.iidm_reg_3 = torch.ones(size=shape, dtype=torch.float, device=self.device)
        # self.iidm_reg_4 = torch.ones(size=shape, dtype=torch.float, device=self.device)
        # self.cah_reg_1 = torch.ones(size=shape, dtype=torch.float, device=self.device)
        # self.cah_reg_2 = torch.ones(size=shape, dtype=torch.float, device=self.device)
        # self.z = torch.ones(size=shape, dtype=torch.float, device=self.device)

        self.AB_CONST = self.ONE * torch.sqrt(2 * hp['a_max'] * hp['b_comf'])
        self.AF_PWR = (self.ONE*torch.clip((-hp['delta']*hp['a_max'] / hp['b_comf']), -4, 0)).to(torch.int)
        
        
    def plot_computed(self, axs, s, show=False):
        """
        Plot computed values of 
        """
        assert(len(axs) >= 6)
        
        self.plot_t_X(axs[0], s,'t','Optimal Distance (m)', self.s_opt, gt=self.s)
        self.plot_t_X(axs[1], s,'t','CAH Acc', self.a_cah)
        self.plot_t_X(axs[2], s,'t','IIDM Acc', self.a_iidm)
        # self.plot_t_X(axs[3], s,'t','IIDM Reg 1', self.iidm_reg_1)
        # self.plot_t_X(axs[4], s,'t','IIDM Reg 2', self.iidm_reg_2)
        # self.plot_t_X(axs[5], s,'t','IIDM Reg 3', self.iidm_reg_3)
        # self.plot_t_X(axs[6], s,'t','IIDM Reg 4', self.iidm_reg_4)
        # self.plot_t_X(axs[7], s,'t','CAH Reg 1', self.cah_reg_1)
        # self.plot_t_X(axs[8], s,'t','CAH Reg 2', self.cah_reg_2)
        # self.plot_t_X(axs[9], s,'t','S opt / Spred', self.z)