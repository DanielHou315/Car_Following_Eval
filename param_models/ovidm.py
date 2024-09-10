"""
This is the abstract class
"""
import os
from math import *
# import numpy as np
import torch
import matplotlib.pyplot as plt
from .param_base_model import *
from .idm import *


OVIDM_def_properties = HyperparamCollection(
    [
        Hyperparam("a_max", HP_REAL, 1.064, 0.01, 5.0), 
        Hyperparam("b_comf", HP_REAL, 0.5, 0.01, 5.0), 
        Hyperparam("s_0", HP_REAL, 4.2995190, 0.1, 10.0), 
        Hyperparam("T", HP_REAL,2.0536119, 0.1, 5.0),  
        Hyperparam("v_opt", HP_REAL, 40.0, 3.0, 60.0),
        Hyperparam("v_mult", HP_REAL, 0.01, 0.01, 5.0),
        Hyperparam("ds", HP_REAL, 12.2, 1.0, 30.0),
        Hyperparam("beta", HP_REAL, 0.389, 0.001, 10.0)
    ]
)

# OVIDM_overall_rmsne_properties = HyperparamCollection(
#     [
#         Hyperparam("a_max", HP_REAL, 1.064, 0.01, 5.0), 
#         Hyperparam("b_comf", HP_REAL, 0.5, 0.01, 5.0), 
#         Hyperparam("s_0", HP_REAL, 4.2995190, 0.1, 10.0), 
#         Hyperparam("T", HP_REAL,2.0536119, 0.1, 5.0),  
#         Hyperparam("v_opt", HP_REAL, 40.0, 3.0, 60.0),
#         Hyperparam("v_mult", HP_REAL, 0.01, 0.01, 5.0),
#         Hyperparam("ds", HP_REAL, 12.2, 1.0, 30.0),
#         Hyperparam("beta", HP_REAL, 0.389, 0.001, 10.0)
#     ]
# )

# OVIDM_overall_mse_properties = HyperparamCollection(
#     [
#         Hyperparam("a_max", HP_REAL, 1.064, 0.01, 5.0), 
#         Hyperparam("b_comf", HP_REAL, 0.5, 0.01, 5.0), 
#         Hyperparam("s_0", HP_REAL, 4.2995190, 0.1, 10.0), 
#         Hyperparam("T", HP_REAL,2.0536119, 0.1, 5.0),  
#         Hyperparam("v_opt", HP_REAL, 40.0, 3.0, 60.0),
#         Hyperparam("v_mult", HP_REAL, 0.01, 0.01, 5.0),
#         Hyperparam("ds", HP_REAL, 12.2, 1.0, 30.0),
#         Hyperparam("beta", HP_REAL, 0.389, 0.001, 10.0)
#     ]
# )

OVIDM_rmsnelb_properties = HyperparamCollection(
    [
        Hyperparam("a_max", HP_REAL, 1.91310542, 0.01, 5.0), 
        Hyperparam("b_comf", HP_REAL, 0.01, 0.01, 5.0), 
        Hyperparam("s_0", HP_REAL,0.676515203, 0.1, 10.0), 
        Hyperparam("T", HP_REAL,0.952044927, 0.1, 5.0),  
        Hyperparam("v_opt", HP_REAL, 60.0, 3.0, 60.0),
        Hyperparam("v_mult", HP_REAL, 0.01, 0.01, 5.0),
        Hyperparam("ds", HP_REAL, 2.01282859, 1.0, 30.0),
        Hyperparam("beta", HP_REAL, 2.72106864, 0.001, 10.0)
    ]
)

OVIDM_rmsneub_properties = HyperparamCollection(
    [
        Hyperparam("a_max", HP_REAL, 2.65829363, 0.01, 5.0), 
        Hyperparam("b_comf", HP_REAL, 0.01, 0.01, 5.0), 
        Hyperparam("s_0", HP_REAL, 0.753168918, 0.1, 10.0), 
        Hyperparam("T", HP_REAL,2.81565046, 0.1, 5.0),  
        Hyperparam("v_opt", HP_REAL, 52.5871238, 3.0, 60.0),
        Hyperparam("v_mult", HP_REAL, 4.59866384, 0.01, 5.0),
        Hyperparam("ds", HP_REAL, 2.96318818, 1.0, 30.0),
        Hyperparam("beta", HP_REAL, 3.74053509, 0.001, 10.0)
    ]
)

OVIDM_mselb_properties = HyperparamCollection(
    [
        Hyperparam("a_max", HP_REAL, 1.81836653, 0.01, 5.0), 
        Hyperparam("b_comf", HP_REAL, 3.24013602, 0.01, 5.0), 
        Hyperparam("s_0", HP_REAL, 1.94039658, 0.1, 10.0), 
        Hyperparam("T", HP_REAL, 1.20175945, 0.1, 5.0),  
        Hyperparam("v_opt", HP_REAL, 60.0, 3.0, 60.0),
        Hyperparam("v_mult", HP_REAL, 0.01, 0.01, 5.0),
        Hyperparam("ds", HP_REAL, 3.64532946, 1.0, 30.0),
        Hyperparam("beta", HP_REAL, 2.02778599, 0.001, 10.0)
    ]
)

OVIDM_mseub_properties = HyperparamCollection(
    [
        Hyperparam("a_max", HP_CONST, 0.72108608, 0.01, 5.0), 
        Hyperparam("b_comf", HP_CONST, 0.47131368, 0.01, 5.0), 
        Hyperparam("s_0", HP_CONST,5.19622118, 0.1, 10.0), 
        Hyperparam("T", HP_CONST,2.18997169, 0.1, 5.0),  
        Hyperparam("v_opt", HP_CONST, 20.57596768, 3.0, 60.0),
        Hyperparam("v_mult", HP_CONST,  1.81781935, 0.01, 5.0),
        Hyperparam("ds", HP_CONST, 7.61824465, 1.0, 30.0),
        Hyperparam("beta", HP_CONST, 1.20729566, 0.001, 10.0)
    ]
)

class BatchOVIDM(BatchIDM):
    def __init__(self, dataset, seqs, name="OVIDM", props=OVIDM_def_properties, pred_field=PRED_ACC, mode="gpu", exclude_ids=[], include_ids=None, flatten=False, chunk_len=None, test_ratio=0):
        super(BatchOVIDM, self).__init__(dataset, seqs, name, props, pred_field, mode, exclude_ids, include_ids, flatten, chunk_len, test_ratio)

    def calc_v_opt(self, t, hp):
        v_sym = hp['v_opt'] * (torch.tanh((self.s_pred[...,t]-hp["s_0"])/hp['ds'] - hp['beta']) + torch.tanh(hp['beta'])) / (1 + torch.tanh(hp["beta"]))
        v_pos = v_sym / 2*(1+torch.tanh(hp['beta'])) + hp['v_opt'] * (1-torch.tanh(hp['beta']))
        return v_pos
    