"""
This is the base classesz
"""
import os
import gc
# import pickle
from math import pow, floor, ceil
from itertools import product
from abc import ABC, abstractmethod

from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from skopt import Optimizer
from skopt.space import Real, Integer
# from joblib import Parallel, delayed
import pygad


HP_CONST = 0
HP_REAL = 1
HP_INT = 2

domain_skopt_dict = {
    HP_CONST: None,
    HP_REAL: Real,
    HP_INT:Integer,
}


A_MAX_PHYS = 5.0
B_MAX_PHYS = -6.0


PRED_ACC = 0
PRED_VEL = 1

class Hyperparam:
    def __init__(self
                 , name: str
                 , domain: str
                 , default
                 , min_val=None
                 , max_val=None
                ):
        """
        name: name of variable
        domain: "r" for real, "i" for integer, "c*" (ci, cr) for constant
        def_val: default value
        min_val: min value, required if domain != 'c*'
        max_val: max value, required if domain != 'c*'
        """
        self.name = name
        if domain not in domain_skopt_dict.keys():
            raise ValueError("domain must be one of r, i, ci, cr!")
        self.domain = domain
        self.default = default
        
        if self.domain == HP_CONST:
            self.min_val = self.default
            self.max_val = self.default
            self.range = 0
            return
        
        if max_val < min_val:
            self.domain = HP_CONST
            self.default = (max_val + min_val)/2
            max_val, min_val = min_val, max_val
                
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
    
    def sample(self, num, method="uniform"):
        # Constants
        if self.domain == HP_CONST:
            return [self.default] * num
    
        # Random Uniform search
        if method == "rand_unif":
            if self.domain == HP_INT:
                return np.random.randint(self.min_val, self.max_val, size=num).astype(np.int32).tolist()
            return np.random.uniform(self.min_val+0.01, self.max_val-0.01, size=num).astype(np.float32).tolist()

        # Uniform search
        if method == "uniform":
            gap = self.range/num 
            if self.domain == HP_INT:
                gap = min(ceil(self.range), floor(gap))
            return np.arange(self.min_val+0.01, self.max_val-0.01, gap).astype(np.float32).tolist()
        
        # Other search method found
        raise Exception(f"Search Method {self.method} not supported!")


class HyperparamCollection:
    def __init__(self, l, construct=False):
        if not isinstance(l, list):
            raise TypeError("Param is not of type List!")
        self.params = {}
        
        # If initial parameters
        for param in l:
            if not isinstance(param, Hyperparam):
                raise TypeError("Param is not of type Hyperparam!")
            self.params[param.name] = param
        return
            
    def from_sample(self, num, device, method="uniform"):
        return self.__row2dict(self.__sample_arr(num, method), device)
    
    def from_unique_sample(self, field, defaults, num, device, method="uniform"):
        idx = -1
        v = None
        for i, val in enumerate(self.params.values()):
            if val.name == field:
                idx = i
                v = val
                break
        ls = [defaults.copy() for _ in range(num)]
        sample = val.sample(num, method)
        min_num = min(len(ls), len(sample))
        for j, l in enumerate(ls[:min_num]):
            l[i] = sample[j]
        npls = [np.array(l) for l in ls]
        return self.__row2dict(npls, device)
    
    def to_ga_default(self, num_total):
        num_each = ceil(pow(num_total, 1/self.num_params))
        return self.__sample_arr(num_each, 'rand_unif')[:num_total]
    
    def __sample_arr(self, num_each, method):
        # Get the sampled parameters and combine them
        sampled_params = [p.sample(num_each, method) for p in self.params.values()]
        # Concatenate and reshape
        ts = [np.array(p, dtype=np.float32) for p in product(*sampled_params)]
        return ts
    
    def to_ga_space(self):
        rtn = []
        for v in self.params.values():
            rtn.append({'low':v.min_val, 'high': v.max_val})
        return rtn
    
    def from_default(self, device):
        rtn = {}
        # Assign names
        for key, v in self.params.items():
            arr = [v.default]
            rtn[key] = torch.tensor(arr, device=device, dtype=torch.float).reshape(1,-1)
        return rtn
    
    def from_array(self, l, device):
        """ 
        Produces a single instance from a pre-existing array
        """
        rtn = {}
        for i, key in enumerate(self.params.keys()):
            rtn[key] = torch.tensor(l[i],device=device).reshape(1,-1)
        return rtn
    
    def from_arrayBF(self, arr: np.ndarray, device):
        """ 
        Produces a param dict from a BatchxnField array
        """
        assert(arr.shape[1] == self.num_params)
        return self.__2Darr2dict(arr, device)
    
    def from_arrayFC(self, arr: torch.Tensor, device):
        """ 
        Produces a param dict from a nCFID x nField array
        """
        rtn = {}
        for i, (key, _) in enumerate(self.params.items()):
            sl = arr[:,i:i+1].reshape(1,-1).to(device)
            rtn[key] = sl
        return rtn
    
#     def from_skopt(self, input_list, device):
#         """Return an hp dict comprehensible by """
#         num_trials = len(input_list)
#         pm_list = []
#         li = 0
#         for k, v in self.params.items():
#             if v.domain == HP_CONST: 
#                 arr = np.array([v.default] * num_trials).astype(np.float32)
#             else: 
#                 arr = np.array([p[li] for p in input_list]).astype(np.float32)
#                 li += 1
#             pm_list.append(arr)
#         return self.__row2dict(pm_list, device)
    
#     def to_skopt(self):
#         """return an initializer list comprehensible by skopt"""
#         rtn = []
#         for p in self.params.values():
#             if p.domain == HP_CONST: 
#                 continue
#             rtn.append(domain_skopt_dict[p.domain](p.min_val, p.max_val))
#         return rtn
                
#     def refine_range(self, l, shrink_factor=2.0):
#         new_init_list = []
#         for i, (key, val) in enumerate(self.params.items()):
#             # Append constants
#             if val.domain == HP_CONST:
#                 p = val
#             # Otherwise, refine search range
#             elif val.domain == HP_INT:
#                 p = Hyperparam(val.name, val.domain,l[i]
#                                , floor(l[i]-val.range/(shrink_factor*2))
#                                , ceil(l[i]+val.range/((shrink_factor*2)))
#                               )
#             else:
#                 p = Hyperparam(val.name, val.domain, l[i]
#                                , l[i]-val.range/(shrink_factor*2)
#                                , l[i]+val.range/(shrink_factor*2)
#                             )
#             new_init_list.append(p)
#         return HyperparamCollection(new_init_list)
    
    def __row2dict(self, rows, device):
        """
        list of parameter entries table (nBatch[nField]) to hp dict for simulation
        
        rows accepted as list of np.ndarray
        """
        ts_all = torch.stack([torch.from_numpy(row) for row in rows])
        return self.__2Darr2dict(ts_all, device)
        
    def __2Darr2dict(self, arr, device):
        """
        2D table (nBatch x nField) to hp dict for simulation
        
        arr as 2D np.ndarray or torch.Tensor accepted
        """
        assert(arr.shape[1] == self.num_params)
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        assert(isinstance(arr, torch.Tensor))
        rtn = {}
        for i, (key, _) in enumerate(self.params.items()):
            sl = arr[:,i:i+1].reshape(-1,1).to(device)
            rtn[key] = sl
        return rtn
    
    @property
    def global_lower_bound(self):
        return min([v.min_val for v in self.params.values()])
    
    @property
    def global_upper_bound(self):
        return max([v.max_val for v in self.params.values()])
    
    @property
    def num_params(self):
        return len(self.params)
    
#     def num_sample_limit(self, indiv_num):
#         total = 1
#         for val in self.params.values():
#             if val.domain != HP_CONST:
#                 total *= indiv_num
#         return total
    
    def __str__(self):
        out = f"Hyperparameter Collection:\n"
        for hp in self.params.values():
            if hp.domain == HP_CONST:
                out += f"- Constant aram {hp.name}={hp.default}\n"
            else:
                out += f"Param {hp.name} with default={hp.default}, range={hp.range}\n"
        return out
        
        
class ParamBaseModelBatch(ABC):
    id_fields = {
        "driver":"driver"
        , "trip":"trip"
        , "congestion":"congestion"
        , "cf_idx":"cf_idx"
        , "type":"type"
    }
        
    data_fields = { "time": "t_st"                     # time since trip
               , "speed":"v"
               , "distance": "x_rough"
               , "accelpedal": "a_pedal"
            #    , "Ax":"a_original"
               , "calibratedAx":"a"
               , "range": "s"
               , "rangerate":"relv"
               # , "timeHeadway":"t_headway"
               # , "TTC":"ttc"
               # , "age":"age"
               # , "gender":"gender"
               , "brakepedal":"b_pedal"
            #    , "pedalMode":"pedal_mode"
            #    , "pedalChange":"pedal_change"
               , "timeSinceCongestion": "t_sc"
               # , "VL_theta": "vl_theta"
               # , "VL_dtheta": "vl_dtheta"
               , "lv_speed": "lv_v"
               , "lv_Ax": "lv_a"
               , "distanceCalibrated": "x"
               , "lv_distance": "lv_x"
               , "timeSinceCF": "t_scf"
               , "distanceSinceCF": "x"
               , "lv_distanceSinceCF": "lv_x"
    }
    
    plot_label_dict = {
        "t": "Time (s)",
        "a": "Acceleration (m/s^2)",
        "v": "Speed (m/s)",
        "x": "Position (m)",
        "s": "Gap to Lead Vehicle (m)",
        "relv": "Relative Speed (m/s)",
    }
    
    TICK = 0.1
    
    MODE_CPU = 0
    MODE_GPU = 1
    MODE_METAL_GPU = 2
    def __init__(self, 
                 dfs, 
                 name: str, 
                 props: dict, 
                 pred_field=PRED_ACC,
                 mode='gpu', 
                 exclude_ids=[], 
                 include_ids=None, 
                 flatten=False,
                 chunk_len=None, 
                 test_ratio=0):
        """
        Initialize the model with given parameters.
        """
        self.model_name = name
        if mode=='gpu':
            self.mode=self.MODE_GPU
            self.device = torch.device('cpu')
        elif mode=='mps':
            self.mode=self.MODE_METAL_GPU
            self.device = torch.device('mps')
        else:
            self.mode=self.MODE_CPU
            self.device = torch.device('cpu')
        self.hyperparams = props
        self.pred_field = pred_field
        assert (pred_field == PRED_ACC) or (pred_field == PRED_VEL)
        
        # Skip those explicitly asked to skip
        if not isinstance(dfs, list):
            dfs = [dfs]
        self.dfs = [df for df in dfs if 
                        (df.loc[df.index[0],"cf_idx"] not in exclude_ids)
                    ]
        # Include those only included
        if include_ids is not None and len(include_ids) > 0 :
            dfs_tmp = [df for df in self.dfs if (df.loc[df.index[0],"cf_idx"] in include_ids)]
            self.dfs = dfs_tmp
        
        # set sim related vars
        self.computed_fields = 0
        self.last_batch_size = -1
        
        # If flatten
        self.flatten = flatten
        if flatten is True:
            # Compute metadata
            self.mat_shape = (1, sum([(len(df)-1) for df in self.dfs]), 2)
            self.start_sim_time = 0
            self.end_sim_time = 1
            self.seq_lens = torch.ones((1, self.mat_shape[1], 1), device=self.device) * 2
            # Initialize matrices for simulation, and that's it
            self.init_flat_matrices(self.dfs)
            return
    
        # elif chunk-ify
        if chunk_len is not None:
            dfs_tmp = []
            for df in self.dfs:
                for i in range(0, len(df), chunk_len):
                    if i+chunk_len*1.1 >= len(df):
                        dfs_tmp.append(df.loc[df.index[i]:, :])
                        break
                    else:
                        dfs_tmp.append(df.loc[df.index[i]:df.index[i+chunk_len-1], :])
            self.dfs = dfs_tmp
            
        self.chunked = (chunk_len is not None)
        self.init_ids(self.dfs)
        
        # Compute metadata
        self.mat_shape = (1, len(self.seqs), max([len(df) for df in self.dfs]))
        
        self.start_sim_time = 0
        self.end_sim_time = self.mat_shape[-1]-1
        
        # Initialize more stuff
        self.init_id_fields(self.dfs)
        self.init_matrices(self.dfs)
        
    def make_dirs(self, root):
        # Make directories
        # for (d, sd) in [(os.path.join(root, (sd)),sd) for sd in ["graph", "cache"]]:
        if not os.path.isdir(root):
            os.makedirs(root)
        
    def init_ids(self, dfs):
        if self.flatten:
            raise Exception("Flattened Model Cannot Init IDs!")
        self.seqs = []
        for df in dfs:
            cfx = df.loc[df.index[0],"cf_idx"]
            self.seqs.append(cfx)
            
    def init_id_fields(self, dfs):
        if self.flatten:
            raise Exception("Flattened Model Cannot Init ID Fields!")
        # Set ID fields and metadata
        self.seq_lens = torch.tensor([len(df) for df in dfs], device=self.device).reshape(1,-1,1)
        for field in ["driver", "trip", "congestion", "cf_idx", "type"]:
            setattr(self, field, [df.loc[df.index[0], field] for df in dfs])
        self.t0 = [df.loc[df.index[0],"timeSinceCongestion"] for df in dfs]

    def init_matrices(self, dfs):
        if self.flatten:
            raise Exception("Flattened Model Cannot Init Matrices, Use init_flat_matrices instead!")
        # Init fields from data
        for field, attr in self.data_fields.items():
            arr = torch.zeros(size=self.mat_shape, dtype=torch.float32, device=self.device)
            idx = 0
            for i, df in enumerate(dfs):
                arr[...,idx,:self.seq_lens[0,i,0]] = self.col2torch(df, field, device=self.device)
                idx += 1
            setattr(self, attr, arr)
            
        # Create a mask
        mask = torch.zeros(size=self.mat_shape, dtype=torch.bool, device=self.device)
        idx = 0
        for i, df in enumerate(dfs):
            mask[..., idx, :self.seq_lens[0,i,0]] = True
            idx += 1
        self.mask = mask
        
    def init_flat_matrices(self, dfs):
        s_idx = 0
        # Init fields from data
        for field, attr in self.data_fields.items():
            arr = torch.zeros(size=self.mat_shape, device=self.device)
            for df in dfs:
                orig_arr = self.col2torch(df, field, device=self.device).reshape(1, -1, 1)
                slen = orig_arr.numel()-1
                arr[:,s_idx:s_idx+slen, 0] = orig_arr[:, :-1, 0]
                arr[:,s_idx:s_idx+slen, 1] = orig_arr[:, 1:, 0]
            setattr(self, attr, arr)
            
        # Create a mask
        self.mask = torch.ones_like(self.a, device=self.device)
        
    @staticmethod
    def col2torch(df, col, device) -> torch.tensor:
        """pd column to shared numpy array"""
        return torch.tensor(df.loc[:, col].values, device=device)
    
    def get_seqs(self, num_samples=0, method="sequential"):
        # Set models
        if self.flatten:
            raise Exception("Flattened Model Cannot Get Seqs!")
            
        choices = None
        if method == "sequential" and num_samples > 0:
            choices = [i for i in range(num_samples)]
        elif method == "random" and num_samples > 0:
            choices = np.random.randint(0, self.min_seq_len, num_samples).tolist()
            
        if choices is not None: 
            return [self.seqs[i] for i in choices]
        return np.arange(self.seq_lens.numel())
    
    # ----------------------------------------
    # 
    # EVALUATION FUNCTIONS
    # 
    # ----------------------------------------
    @staticmethod
    def __masked_RMSE(pred, gt, sl, mask, axis=(-1)):
        """return RMSE of two arrays"""
        gt = gt + 0.001*(gt==0)
        diff2 = (mask*(pred-gt)**2)
        return torch.sqrt((diff2).sum(axis=axis) / sl.sum(axis=axis))
    
    @staticmethod
    def __masked_RMSNE(pred, gt, sl, mask, axis=(-1)):
        """return RMSNE of two arrays, summed over Sequence and Time"""
        # Make sure divisor is non zero
        gt = gt + 0.001*(gt==0)
        # Compute difference, and clip to 1e6, so that when summed up, result is reasonable
        diff = torch.clip(mask * torch.abs(pred-gt)/gt, 0, 1e9)
        rmsne = torch.sqrt((diff**2).sum(axis=axis) / sl.sum(axis=axis))
        return rmsne
    
    @staticmethod
    def __masked_RMSNE_LB(pred, gt, sl, mask, axis=(-1)):
        """return RMSNE for Lower Bound of two arrays, summed over Sequence and Time"""
        # Make sure divisor is non zero
        gt = gt + 0.001*(gt==0)
        # Compute difference, and clip to 1e6, so that when summed up, result is reasonable
        diff = torch.clip(mask * (pred-gt)/gt, -1e9, 1e9)
        # Penalize 15x much more if pred > gt
        diff = torch.abs(diff + diff * 4 * mask*(pred > gt))
        rmsne = torch.sqrt((diff**2).sum(axis=axis) / sl.sum(axis=axis))
        return rmsne
    
    @staticmethod
    def __masked_RMSNE_UB(pred, gt, sl, mask, axis=(-1)):
        """return RMSNE for Upper Bound of two arrays, summed over Sequence and Time"""
        # Make sure divisor is non zero
        gt = gt + 0.001*(gt==0)
        # Compute difference, and clip to 1e6, so that when summed up, result is reasonable
        diff = torch.clip(mask * (pred-gt)/gt, -1e9, 1e9)
        # Penalize 5x much more if pred < gt
        diff = torch.abs(diff + diff * 4 * mask*(pred < gt))
        rmsne = torch.sqrt((diff**2).sum(axis=axis) / sl.sum(axis=axis))
        return rmsne
    
    @staticmethod
    def masked_NAE(pred, gt, mask):
        """return RMSNE of two arrays, summed over Sequence and Time"""
        # Make sure divisor is non zero
        gt = gt + 0.001*(gt==0)
        # Compute difference, and clip to 1e6, so that when summed up, result is reasonable
        diff = torch.clip(mask * torch.abs(pred-gt)/gt, 0, 1e9)
        return diff
    
    @staticmethod
    def __masked_MSE(pred, gt, sl, mask, axis=(-1)):
        """return RMSE of two arrays"""
        gt = gt + 0.001*(gt==0)
        diff2 = torch.clip(mask*(pred-gt)**2, 0, 1e9)
        mse = (diff2).sum(axis=axis) / sl.sum(axis=axis)
        return mse
    
    @staticmethod
    def __masked_MSE_LB(pred, gt, sl, mask, axis=(-1)):
        """return RMSE of two arrays"""
        gt = gt + 0.001*(gt==0)
        diff = pred-gt
        diff = diff + diff*3*(pred > gt)
        diff2 = torch.clip(mask*(diff)**2, 0, 1e9)
        mse = (diff2).sum(axis=axis) / sl.sum(axis=axis)
        return mse
    
    @staticmethod
    def __masked_MSE_UB(pred, gt, sl, mask, axis=(-1)):
        """return RMSE of two arrays"""
        gt = gt + 0.001*(gt==0)
        diff = pred-gt
        diff = diff + diff*3*(pred < gt)
        diff2 = torch.clip(mask*(diff)**2, 0, 1e9)
        mse = (diff2).sum(axis=axis) / sl.sum(axis=axis)
        return mse
    
    
    def init_sim_matr(self, batch_size, hp):
        assert(batch_size > 0)
        if batch_size == self.last_batch_size:
            return None
        shape = (batch_size, self.mat_shape[1], self.mat_shape[2])
        self.last_batch_size = batch_size
        
        self.mask_batch = self.mask[0:1,...].repeat(batch_size,1,1)  # 0:1 is a hack to keep dim
        
        # Initialize, this is copied for calibration purpose
        self.a_pred = torch.ones(size=shape, dtype=torch.float, device=self.device)
        self.a_pred[...,self.start_sim_time] = self.a[...,self.start_sim_time]
        self.v_pred = torch.ones(size=shape, dtype=torch.float, device=self.device)
        self.v_pred[...,self.start_sim_time] = self.v[...,self.start_sim_time]
        self.s_pred = torch.ones(size=shape, dtype=torch.float, device=self.device)
        self.s_pred[...,self.start_sim_time] = self.s[...,self.start_sim_time]
        self.x_pred = torch.ones(size=shape, dtype=torch.float, device=self.device)
        self.x_pred[...,self.start_sim_time] = self.x[...,self.start_sim_time]
        self.relv_pred = torch.ones(size=shape, dtype=torch.float, device=self.device)
        self.relv_pred[...,self.start_sim_time] = self.relv[...,self.start_sim_time]
        
        self.ZERO = torch.zeros(size=(batch_size, self.mat_shape[1]),device=self.device)
        self.ONE = torch.ones(size=(batch_size, self.mat_shape[1]),device=self.device)
        return shape

    def simulate_step(self, t, hp):
        if self.pred_field == PRED_ACC:
            self.a_pred[...,t] = self.calc_a(t, hp)
            # Clip velocity output to non-negative values
            self.v_pred[...,t+1] = torch.maximum(self.ZERO, self.TICK * self.a_pred[...,t] + self.v_pred[...,t])
            # Clip acceleration values to 0 if velocity was clipped
            self.a_pred[...,t] = self.a_pred[...,t] * torch.logical_not(torch.logical_and((self.v_pred[...,t]==self.ZERO),  self.v_pred[...,t+1]==self.ZERO))
            self.x_pred[...,t+1] = self.x_pred[...,t] + self.v_pred[...,t] * self.TICK + self.a_pred[...,t] * self.TICK**2 / 2.0
            self.s_pred[...,t+1] = self.lv_x[...,t+1] - self.x_pred[...,t+1]
            self.relv_pred[...,t+1] = self.lv_v[...,t+1] - self.v_pred[...,t+1]
        elif self.pred_field == PRED_VEL:
            self.v_pred[..., t] = self.calc_v(t, hp)
            self.a_pred[..., t] = self.TICK * 2 * (self.v_pred[..., t] - self.v_pred[..., t-2])   # Average acceleration
            self.x_pred[...,t] = self.x_pred[...,t-1] + self.v_pred[...,t] * self.TICK + self.a_pred[...,t] * self.TICK**2 / 2.0
            self.s_pred[...,t] = self.lv_x[...,t] - self.x_pred[...,t]
            self.relv_pred[...,t] = self.lv_v[...,t] - self.v_pred[...,t]
        else:
            raise Exception("Unknown Prediction Parameter!")
    
    def post_sim(self):
        if self.pred_field == PRED_ACC:
            self.a_pred[...,self.end_sim_time:] = self.a_pred[...,self.end_sim_time-1:self.end_sim_time]
            if self.end_sim_time != self.mat_shape[-1]-1:
                self.v_pred[...,self.end_sim_time+1:] = self.v_pred[...,self.end_sim_time:self.end_sim_time+1]
                self.x_pred[...,self.end_sim_time+1:] = self.x_pred[...,self.end_sim_time:self.end_sim_time+1]
                self.s_pred[...,self.end_sim_time+1:] = self.s_pred[...,self.end_sim_time:self.end_sim_time+1]
                self.relv_pred[...,self.end_sim_time+1:] = self.relv_pred[...,self.end_sim_time:self.end_sim_time+1]
        elif self.end_sim_time != self.mat_shape[-1]-1:
            self.a_pred[...,self.end_sim_time:] = self.a_pred[...,self.end_sim_time-1:self.end_sim_time]
            self.v_pred[...,self.end_sim_time:] = self.v_pred[...,self.end_sim_time-1:self.end_sim_time]
            self.x_pred[...,self.end_sim_time:] = self.x_pred[...,self.end_sim_time-1:self.end_sim_time]
            self.s_pred[...,self.end_sim_time:] = self.s_pred[...,self.end_sim_time-1:self.end_sim_time]
            self.relv_pred[...,self.end_sim_time:] = self.relv_pred[...,self.end_sim_time-1:self.end_sim_time]

    def simulate(self, hp, batch_size=1, pbar=False):
        # Initialize, this is copied for calibration purpose
        self.init_sim_matr(batch_size, hp)
        
        rg = range(self.start_sim_time, self.end_sim_time)
        if pbar: 
            rg = tqdm(rg)
        
        # Simulate for each step
        with torch.no_grad():
            for t in rg:
                self.simulate_step(t, hp)
        self.post_sim()
    # ----------------------------------------
    # 
    # OPTIMIZATION
    # 
    # ----------------------------------------
    
    
    # def calibrate_scipy():
        # The old scipy optimization, not really working
        # res = minimize(self.benchmark, list(self.hyperparams.values()), tol=1e-1)
        
#     def skopt_calibrate_split(self, batch_size, objective, iters=1):
#         # VERY SLOW STILL
#         def_par_list = self.hyperparams.to_skopt()
#         jam_optimizer = Optimizer(
#             dimensions=def_par_list,
#             random_state=42,
#             base_estimator='gp'
#         )
#         pre_optimizer = Optimizer(
#             dimensions=def_par_list,
#             random_state=42,
#             base_estimator='gp'
#         )
#         post_optimizer = Optimizer(
#             dimensions=def_par_list,
#             random_state=42,
#             base_estimator='gp'
#         )
        
#         for i in tqdm(range(iters)):
#             jam_x = jam_optimizer.ask(n_points=batch_size)
#             pre_x = pre_optimizer.ask(n_points=batch_size)
#             post_x = post_optimizer.ask(n_points=batch_size)
            
            
#             param_batch = self.construct_batch_param(batch_size
#                                                      , (batch_size, self.seq_lens.numel())
#                                                      , self.hyperparams.from_skopt(jam_x, None, self.device)
#                                                      , self.hyperparams.from_skopt(pre_x, None, self.device)
#                                                      , self.hyperparams.from_skopt(post_x, None, self.device))[0]
#             self.simulate(batch_size=64, hp=param_batch)
#             stats = self.compute_stats(objective)
            
#             jam_optimizer.tell(jam_x, stats["jam_"+objective].detach().cpu().tolist())
#             pre_optimizer.tell(pre_x, stats["pre_"+objective].detach().cpu().tolist())
#             post_optimizer.tell(post_x, stats["post_"+objective].detach().cpu().tolist())
        
#         for name, opt in zip(["jam", "pre", "post"], [jam_optimizer, pre_optimize, post_optimizer]):
#             print(f"{self.model_name} {name} minimized at:", min(opt.yi), \
#                   "with stats", opt.Xi[np.argmin(np.array(opt.yi))])  # print the best objective found
    
    def __ga_fitness_func(self, ga_instance, solution, solution_idx): 
        # USED ONLY FOR GENETIC ALGORITHM CALIBRATION, NO OTHER PURPOSES
        assert(getattr(self, "ga_calib_objective") is not None)
        
        # Prepare simulation
        batch_size = solution.shape[0]
        hp = self.hyperparams.from_arrayBF(solution, self.device)
        
        # Simulate and Compute Stats
        self.simulate(hp, batch_size)
        stats = self.compute_stats(self.ga_calib_objective)
        loss = np.exp(-stats["overall_"+self.ga_calib_objective].detach().cpu().numpy())
        return loss
    
    def __ga_on_generation(self, ga_instance):
        self.ga_progress.update(1)
        
    
    def ga_calibrate(self, objective, sol_size=512, batch_size=64, num_generations=16):
        """
        Calibrate a specific target using genetic algorithm
        
        NOTE: this optimizer does not support multi-goal calibration the way I made other ones work. 
        
        objective: "s_ego_pred_rmse" as stats require
        """
        # Setup Goals
        self.ga_calib_objective = objective
        print(objective)
        
        # Setup progress bar
        self.ga_progress = tqdm(total=num_generations, desc='Generations Progress')
        batch_size = min(sol_size, batch_size)
    
        print(f"GA Optimizing with batch size {batch_size}")
        
        g_min = self.hyperparams.global_lower_bound
        g_max = self.hyperparams.global_upper_bound
        gene_space = self.hyperparams.to_ga_space()
        
        # Make an instance
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=4,
            initial_population=self.hyperparams.to_ga_default(sol_size),
            sol_per_pop=sol_size,
            num_genes=self.hyperparams.num_params,
            fitness_batch_size=batch_size,
            random_mutation_min_val=g_min,
            random_mutation_max_val=g_max,
            gene_space=gene_space,
            fitness_func=self.__ga_fitness_func,
            on_generation=self.__ga_on_generation,
            parent_selection_type="sss",
            keep_parents=1,
            crossover_type="two_points",
            mutation_type="random"
        )

        # Run optimization
        ga_instance.run()
        
        # Report results
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"GA found minimized {self.ga_calib_objective} for {self.model_name} at {-np.log(solution_fitness)}")
        print(f"- Parameters found: {solution}")
        
        del ga_instance
        torch.cuda.empty_cache()

        # End
        self.ga_progress.close()
        return solution
        

            
    def grid_calibrate(self, params, objective="a_ego_pred_rmse", batch_size=32):
        """
        objective must be one of: things returned by stats
        cal_dict has fields: overall, (jam, pre, post, indiv) + objective, where each itself and _argmin
        
        - params: HyperParamCollection.sample()/default() output dictionary
        - objective: a_ego_pred_rmse, ..._rmse
        - split_optimize: optimize separately for jam, pre-jam, post-jam if True
        - batch_size: size of batch to compute on GPU
        - opt_iters: iterations to search for loweset and construct new ranges
        - load_record: try to load existing lowest records from 
        """
        if self.flatten:
            raise Exception("Flattened Model Cannot Optimize!")
            
        # Setup
        torch.cuda.empty_cache()
        
        # Loading makes no sense anymore if sampling is random
        self.simulate(self.hyperparams.from_default(device=self.device), batch_size=1)
        cal_dict = self.compute_stats(objective)
        
        # Initialize jam params
        num_fields = self.hyperparams.num_params
        num_samples = next(iter(params.values())).numel()

        with torch.no_grad():
            # Create batches from samples
            param_batches = self.construct_batch_param(num_samples, (batch_size,self.seq_lens.numel()), params)
            num_batches = len(param_batches)
            
            # Run optimization loop
            for batch_idx in range(0, num_batches):
                # Compute Metadata
                abs_idx = batch_idx*batch_size
                b_size = min(batch_size, num_samples-batch_idx*batch_size)

                # Simulate
                self.simulate(param_batches[batch_idx], b_size)
                stats = self.compute_stats(objective)

                # If better, update
                m = torch.min(stats["overall_"+objective])
                if stats["overall_"+objective+"_min"] < cal_dict["overall_"+objective+"_min"]:
                    cal_dict["overall_"+objective+"_min"] = stats["overall_"+objective+"_min"]
                    cal_dict["overall_"+objective+"_argmin"] = (stats["overall_"+objective+"_argmin"]+abs_idx).to(torch.int)

                # Compute binary array for whether smaller is achieved
                is_new_smaller = (stats[objective+"_min"] < cal_dict[objective+"_min"])
                # Update lowest loss and index of corresponding
                cal_dict[objective+"_min"] = (torch.logical_not(is_new_smaller)*cal_dict[objective+"_min"] \
                                    + is_new_smaller*(stats[objective+"_min"]))
                cal_dict[objective+"_argmin"] = torch.logical_not(is_new_smaller)*cal_dict[objective+"_argmin"] \
                                    + is_new_smaller*(stats[objective+"_argmin"]+abs_idx).to(torch.int)

        param_arr = torch.stack([v for v in params.values()],axis=1).reshape(num_samples,num_fields)
        
        cal_dict["overall_"+objective+"_opt_params"] = param_arr[cal_dict["overall_"+objective+"_argmin"], :]
        cal_dict[objective+"_opt_params"] = param_arr[cal_dict[objective+"_argmin"], :]
        
        # Assign all latest
        torch.cuda.empty_cache()

        # Report Iteration
        # print("- optimal:", cal_dict["overall_"+objective+"_min"].item()
        #           , "with params", cal_dict["overall_"+objective+"_opt_params"].detach().cpu().tolist(), '\n')

        # End calibration
        return cal_dict

    def construct_batch_param(self, num_samples, BxN_size, param):
        if self.flatten:
            raise Exception("Flattened Model Cannot Optimize!")
            
        rtn = []
        for s in range(0, num_samples, BxN_size[0]):
            b_size = min(BxN_size[0], num_samples-s)
            p = {}
            for key, val in param.items():
                arr = val[s:s+b_size,:].reshape(b_size,-1)
                p[key] = arr
            rtn.append(p)
        return rtn
    
    
    def per_param_optimize(self, global_opt_params, objective):
        if self.flatten:
            raise Exception("Flattened Model Cannot Optimize!")
            
        def_opt_params = self.hyperparams.from_array(global_opt_params, self.device)
        self.simulate(def_opt_params, batch_size=1)
        cal_dict = self.compute_stats(objective)
        min_vals = [cal_dict[objective+"_min"]]
        opt_params = [torch.Tensor(global_opt_params).reshape(-1,1).repeat(1, self.mat_shape[1])]
        stats = [(torch.mean(cal_dict[objective+"_min"]), torch.var(cal_dict[objective+"_min"]))]
        for param in tqdm(self.hyperparams.params.values()):
            params = self.hyperparams.from_unique_sample(param.name, global_opt_params, 1024, self.device, method="uniform")
            cal_dict = self.grid_calibrate(params, objective, batch_size=256)
            min_vals.append(cal_dict[objective+"_min"])
            opt_params.append(cal_dict[objective+"_opt_params"])
            stats.append((torch.mean(cal_dict[objective+"_min"]), torch.var(cal_dict[objective+"_min"])))
        rtn = {
            "objective": objective,
            "min_vals": min_vals,
            "opt_params": opt_params,
            "stats": stats
        }
        return rtn
        
    
    
    # ----------------------------------------
    # 
    # COMPUTING STATISTICS
    # 
    # ----------------------------------------
    def compute_stats(self, objective) -> dict:
        """
        Compute statistics of simulation results
        
        sim_res_dict: dict[torch.tensor[BxNxT]]
        """
        if isinstance(objective, list):
            return [self.compute_stats(obj) for obj in objective]
        
        methods = {
            "rmse": self.__masked_RMSE,
            "mse": self.__masked_MSE, 
            "mseub": self.__masked_MSE_UB, 
            "mselb": self.__masked_MSE_LB, 
            "rmsne": self.__masked_RMSNE,
            "rmsneub": self.__masked_RMSNE_UB,
            "rmsnelb": self.__masked_RMSNE_LB,
        }
        
        bench_vals = {
            'a': (self.a, self.a_pred), 
            'v': (self.v, self.v_pred), 
            'x': (self.x, self.x_pred), 
            's': (self.s, self.s_pred), 
            'relv': (self.relv, self.relv_pred),
        }
        
        args = objective.split('_')
        loss = methods[args[-1]]
        (gt, pred) = bench_vals[args[0]]
        
        rtn = {}
            
        assert(torch.isnan(gt).any() == False)
        assert(torch.isnan(pred).any() == False)

        # We want to compute per-batch-cf rmse, per-batch rmse
        rtn[objective] = loss(pred, gt, self.seq_lens, self.mask, axis=(2))
        assert(torch.isnan(rtn[objective]).any() == False)
        
        # per-batch overall
        rtn["overall_"+objective] = loss(pred, gt, self.seq_lens, self.mask,axis=(1,2))
        assert(torch.isnan(rtn["overall_"+objective]).any() == False)
        
        # per-batch-group rmse
        # for name, seq in zip(["jam_","pre_","post_"], [self.jam_seqs, self.pre_seqs, self.post_seqs]):
        #     if seq is not None:
        #         rtn[name+objective] = loss(pred[:,seq,:], gt[:,seq,:], 
        #                                     self.seq_lens[:,seq,:], self.mask[:,seq,:],axis=(1,2))
        #         assert(torch.isnan(rtn[name+objective]).any() == False)
            
        # And also per-cf min, (N)
        rtn[objective+"_min"] = torch.min(rtn[objective], axis=0)[0]
        assert(torch.isnan(rtn[objective+"_min"]).any() == False)
        # overall argmin and per-group min, (1)   
        rtn["overall_"+objective+"_min"] = torch.min(rtn["overall_"+objective])
        # for name, seq in zip(["overall_", "jam_","pre_","post_"], [self.seqs, self.jam_seqs, self.pre_seqs, self.post_seqs]):
        #     if seq is not None:
        #         rtn[name+objective+"_min"] = torch.min(rtn[name+objective])
        #         assert(torch.isnan(rtn[name+objective+"_min"]).any() == False)
        
        # And also per-cf argmin (N) 
        rtn[objective+"_argmin"] = torch.argmin(rtn[objective], axis=0).to(torch.int)
        rtn["overall_"+objective+"_argmin"] = torch.argmin(rtn["overall_"+objective], axis=0).to(torch.int)
        # overall argmin and per-group rmse 
#         for name, seq in zip(["overall_", "jam_","pre_","post_"], [self.seqs, self.jam_seqs, self.pre_seqs, self.post_seqs]):
#             if seq is not None:
#                 rtn[name+objective+"_argmin"] = torch.argmin(rtn[name+objective], axis=0).to(torch.int)
            
        # Collision  
        rtn["num_collisions"] = ((self.s_pred[...,1:] < 0) \
                          & (self.s_pred[...,:-1] >= 0)).sum(axis=-1)                             # BxN
        rtn["num_collisions_argmin"] = torch.argmin(rtn["num_collisions"]).to(torch.int)
        return rtn

    # ----------------------------------------
    # 
    # PLOTTING
    # 
    # ----------------------------------------
    
    # Simulation Plots
    def plot_sim_results(self, seqs, out_dir, label, show=False, debug=False):
        """
        plot simulation results
        
        seqs: sequences to plot
        sim_result: dict from simulate()
        labels: 
        """
        if self.flatten:
            raise Exception("Flattened Model Cannot Graph!")
            
        assert(self.a_pred.size()[0] == 1)
        self.make_dirs(out_dir)
        # Plot overview
        self.plot_sim_overview(label, out_dir, show)
        # Plot individual ones
        num_subplots_per_row = max(5, (self.computed_fields*debug))
        for s in tqdm(seqs):
            self.plot_sim_one(s, num_subplots_per_row, label, out_dir, show, debug)
            
    def plot_sim_overview(self, label, root, show=False):
        if self.flatten:
            raise Exception("Flattened Model Cannot Graph!")
            
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))

        # Plot individual df
        self.plot_scatter(axs[0],'v','s', self.v_pred, self.s_pred, gt_x=self.v, gt=self.s)
        self.plot_scatter(axs[1],'a','s', self.a_pred, self.s_pred, gt_x=self.a, gt=self.s)

        plt.suptitle(f"Summary of {self.model_name}")
        if show: 
            plt.show()
        plt.savefig(os.path.join(root, f"{self.model_name}_overview_{label}.png"))
        plt.close()
            
    def plot_sim_one(self, s, num_subplots_per_row, label, root, show=False, debug=False):
        if self.flatten:
            raise Exception("Flattened Model Cannot Graph!")
            
        num_cols = (1+debug)
        fig, axs = plt.subplots(num_subplots_per_row, num_cols, figsize=(16, 4*num_subplots_per_row))
        
        axs = axs.reshape(num_subplots_per_row, num_cols)
        # Plot individual df
        self.plot_t_X(axs[0,0], s,'t','a', self.a_pred, self.a, self.lv_a)
        self.plot_t_X(axs[1,0], s, 't', 'v', self.v_pred, self.v, self.lv_v)
        self.plot_t_X(axs[2,0], s, 't', 'x', self.x_pred, self.x, self.lv_x)
        self.plot_t_X(axs[3,0], s, 't', 's', self.s_pred, self.s)
        self.plot_t_X(axs[4,0], s, 't', 'relv', self.relv_pred, self.relv)
        
        if debug == True:
            self.plot_computed(axs[:,1], s)

        plt.suptitle(f"{self.model_name} Driver {self.driver[s]}, Trip {self.trip[s]},Time {self.t0[s]:.1f},CF {self.cf_idx[s]}")
        if show: 
            plt.show()
        
        i = 0
        fig_path = os.path.join(root, f"{self.model_name}_{self.cf_idx[s]}_{label}_chunk{i}_sim_result.png")
        while os.path.isfile(fig_path):
            i += 1
            fig_path = os.path.join(root, f"{self.model_name}_{self.cf_idx[s]}_{label}_chunk{i}_sim_result.png")
        plt.savefig(fig_path)
        plt.close()
        gc.collect()
        
    def plot_t_X(self, ax, s, x_label, y_label, pred, gt=None, lv=None):
        if self.flatten:
            raise Exception("Flattened Model Cannot Graph!")
            
        slen = self.seq_lens[0,s].item()
        t = self.t_sc[0, s, :slen].detach().cpu().numpy()
        
        # Plot Pred
        pred = pred[0, s, :slen].detach().cpu().numpy()
        ax.plot(t, pred, label=f'{self.model_name} Prediction')
        
        if gt is not None:
            gt = gt[0, s, :slen].detach().cpu().numpy()
            ax.plot(t, gt, label=f'Ground Truth')
        
        if lv is not None:
            lv = lv[0, s, :slen].detach().cpu().numpy()
            ax.plot(t, lv, label=f'Lead Vehicle')
        
        if x_label in self.plot_label_dict.keys():
            x_label = self.plot_label_dict[x_label]
        if y_label in self.plot_label_dict.keys():
            y_label = self.plot_label_dict[y_label]
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        
    # Per Parameter Optimization Overview
    def plot_per_param_opt_indv(self, opt_dict, best_idx):
        if self.flatten:
            raise Exception("Flattened Model Cannot Graph!")
            
        self.make_dirs("results/"+self.model_name+"_per_param_opt/")
        
        indv_params = self.hyperparams.from_arrayFC(opt_dict["opt_params"][best_idx], self.device)
        self.simulate(indv_params, batch_size=1)
        # stats = self.compute_stats(opt_dict["objective"])
        seqs = self.get_seqs(num_samples=0,groups=["jam"], method="sequential")
        self.plot_sim_results(seqs, out_dir="results/"+self.model_name+"_per_param_opt", label="per_param_best", show=False, debug=False)

    def plot_per_param_opt_overview(self, opt_dict):
        if self.flatten:
            raise Exception("Flattened Model Cannot Graph!")
            
        self.make_dirs("results/"+self.model_name+"_per_param_opt/")
        plt.figure(figsize=(12,8))
        
        min_vals = opt_dict["min_vals"]
        stats =  opt_dict["stats"]
        
        print("Plotting Per-Param Optimize Result")
        plt.boxplot([arr.detach().cpu().numpy() for arr in min_vals], vert=True)
        
        # Add title and labels
        plt.title(f'{self.model_name} Per-Parameter Optimization Result')
        plt.xlabel('Optimization Target')
        plt.ylabel(opt_dict["objective"]+' distribution')
        
        labels = [f"Global Optimum\nM:{stats[0][0]:.2f}, Var:{stats[0][1]:.2f}"]
        labels.extend([lab+f"\nM:{stats[i+1][0]:.2f}, Var:{stats[i+1][1]:.2f}" for i, lab in enumerate(self.hyperparams.params.keys())])
        plt.xticks(range(1, len(labels) + 1), labels)

        # Display the plot
        plt.show()
        plt.savefig("results/"+self.model_name+"_per_param_opt/"+self.model_name+"_per_param_opt_overview.png")

    def plot_scatter(self, ax, x_label, y_label, x_vals, pred, gt_x=None, gt=None, lv=None):
        if self.flatten:
            raise Exception("Flattened Model Cannot Graph!")
            
        """scatter plot of all data aggregated in this model"""
        x = x_vals[self.mask].detach().cpu().numpy()
        pred = pred[self.mask].detach().cpu().numpy()
        ax.scatter(x, pred, s=3, label=f"{self.model_name} Prediction")
        
        if gt is not None:
            gt = gt[self.mask].detach().cpu().numpy()
            gt_x = gt_x[self.mask].detach().cpu().numpy()
            ax.scatter(gt_x, gt, s=1, label="Ground Truth")
            
        if lv is not None:
            lv = lv[self.mask].detach().cpu().numpy()
            ax.scatter(x, lv, s=1, label="Lead Vehicle")
        
        if x_label in self.plot_label_dict.keys():
            x_label = self.plot_label_dict[x_label]
        if y_label in self.plot_label_dict.keys():
            y_label = self.plot_label_dict[y_label]
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
    
    def plot_computed(self, axs, s, show=False):
        """Make a new model!"""
        if self.flatten:
            raise Exception("Flattened Model Cannot Graph!")