"""
This is the abstract class
"""
import os
from math import *
import numpy as np
import matplotlib.pyplot as plt
from .param_base_model import *
from .ovm import BatchOVM


# This says zotero://select/library/items/YBWINB3Q

# afvdm =3.2431 tanh(0.13 dx − 2.22) − 0.41v + 0.2 dv + 2.7675.

# FVDM_def_properties = HyperparamCollection(
#     [
#         Hyperparam("alpha", HP_REAL, 0.41, 0.01, 2.0),
#         Hyperparam("v_ave", HP_REAL, 3.2431/0.41, 1.0, 50.0),
#         Hyperparam("v_mult", HP_REAL, 0.13, 0.01, 5.0),
#         Hyperparam("v_sub", HP_REAL, 2.22, 0.01, 50.0),
#         # Hyperparam("v_bias", HP_REAL, 2.7675, 0.0, 10.0),
#         Hyperparam("v_bias", HP_REAL, 0.913, 0.01, 10.0),
#         Hyperparam("lambda", HP_REAL, 0.2, 0.01, 10.0),
#     ]
# )
FVDM_def_properties = HyperparamCollection(
    [
        Hyperparam("alpha", HP_REAL, 1/5.12, 0.1, 2.0),
        Hyperparam("v_max", HP_REAL, 36.13, 1.0, 40.0),
        Hyperparam("s_0", HP_REAL, 4, 1.0, 30.0),
        Hyperparam("theta", HP_REAL, 9.41, 1.0, 30.0),
        Hyperparam("beta", HP_REAL, 0.10, 0.001, 10.0), 
        Hyperparam("lambda", HP_REAL, 0.2, 0.001, 10.0),
    ]
)


class BatchFVDM(BatchOVM):
    def __init__(self, dataset, name="FVDM", props=FVDM_def_properties, pred_field=PRED_ACC, mode="gpu", exclude_ids=[], chunk_len=None, test_ratio=0):
        super(BatchFVDM, self).__init__(dataset, name, props, pred_field, mode, exclude_ids, chunk_len, test_ratio)
        self.computed_fields = 2
    
    def calc_a(self, t, hp):
        a_ovm = super().calc_a(t, hp)
        return torch.clip(a_ovm + hp['lambda']*self.relv[...,t], self.ONE*B_MAX_PHYS, self.ONE*A_MAX_PHYS)