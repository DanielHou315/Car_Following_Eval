# import torch
# from torch.utils.data import Dataset
# from .DTrainer.dtrainer import DTrainer
# from torch import nn
# from torch.utils.data import DataLoader
# import torch.optim as optim

# from sklearn.model_selection import KFold
# # Ensemble weights optimization (example using least squares)
# from scipy.optimize import minimize

# from .param_base_model import *


# class MoE_Dataset(Dataset):
#     """
#     Dataset for NN_MoE Training
#     """
#     def __init__(self, X, y):
#         """
#         len_sequence: length of each sequence feature
#         average_dist: average distance between sampled starting points
#         label_cnt: take the last how many accelerations as label
#         """
#         self.X = X
#         self.y = y

#     def __getitem__(self, idx):
#         return self.X[idx,:], self.y[idx,:]

#     def __len__(self):
#         return self.X.size()[0]

    
# class MoE_DataMgr:
#     """
#     Dataset for NN_MoE Training
#     """
#     def __init__(self, dataset, models):
#         """
#         len_sequence: length of each sequence feature
#         average_dist: average distance between sampled starting points
#         label_cnt: take the last how many accelerations as label
#         """
#         self.n_models = len(models)
        
#         # Nx3 features
#         self.features = torch.cat([
#                         models[0].v.clone().detach(), 
#                         models[0].s.clone().detach(), 
#                         models[0].relv.clone().detach()
#                        ], axis=2).reshape(-1, 3)
        
#         for mod in models:
#             mod.simulate(mod.hyperparams.from_default(mod.device))
            
#         # Nx(nModel+1) labels
#         outs = [mod.a_pred for mod in models]
#         outs.append(models[0].a)
#         self.labels = torch.cat(outs, axis=2).reshape(-1, self.n_models+1)

#     def __getitem__(self, idx):
#         return self.features[idx,:], self.labels[idx,:]

#     def __len__(self):
#         return self.features.size()[0]

    
# TORCH_MSE = nn.MSELoss()
# def moe_a_loss(w_pred, y):
#     """
#     w_pred: BxnModel
#     y: Bx(nModel+1)
#     """
#     weighted_a = (w_pred * y[...,:-1]).sum(axis=-1)
#     return torch.sqrt(TORCH_MSE(weighted_a, y[..., -1]))


# # class MLP_MoE(nn.Module):
# #     def __init__(self,n_models, n_features):
# #         super(MLP_MoE, self).__init__()
        
# #         self.fc_1 = nn.Linear(n_features, 16)
# #         self.fc_2 = nn.Linear(16, 32)
# #         self.fc_3 = nn.Linear(32, 64)
# #         self.fc_4 = nn.Linear(64, n_models)
# #         self.ReLU = nn.ReLU()
# #         self.Softmax = torch.nn.Softmax(dim=0)
        
# #     def init_weights(self):
# #         for layer in [self.fc_1, self.fc_2, self.fc_3, self.fc_4]:
# #             nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
# #             if layer.bias is not None:
# #                 nn.init.zeros_(layer.bias)

# #     def forward(self, x):
# #         out = x
# #         for layer in [self.fc_1, self.fc_2, self.fc_3]:
# #             out = self.ReLU(layer(out))
# #         out = self.fc_4(out)
# #         return self.Softmax(out)
    
    
# class SoftmaxMoE(nn.Module):
#     def __init__(self,n_models, n_features):
#         super(SoftmaxMoE, self).__init__()
        
#         self.k = torch.zeros(size=(n_features, n_models))
#         self.b = torch.zeros(size=(1, n_models))
        
#     def init_weights(self):
#         for layer in [self.k, self.b]:
#             nn.init.kaiming_normal_(layer, nonlinearity='relu')

#     def forward(self, x):
#         # res in (B)x(nSeq)x(nModel)
#         res = torch.exp(x * k + b)
#         r_sum = res.sum(axis=1)
#         return res / r_sum

    
# MoE_def_properties = HyperparamCollection([])

# class BatchGMMMoE(ParamBaseModelBatch):
#     def __init__(self, name, ensembler_stumps, dataset, gmm, exclude_ids):
#         super(BatchGMMMoE, self).__init__(dataset,
#                  name, 
#                  MoE_def_properties,
#                  pred_field=ensembler_stumps[0].pred_field,
#                  mode="gpu", 
#                  exclude_ids=exclude_ids, 
#                  include_ids=None,
#                  flatten=False,
#                  chunk_len=None, 
#                  test_ratio=0
#         )
#         self.computed_fields=0
#         self.bm = ensembler_stumps[0]
#         self.gmm = gmm
#         self.start_sim_time = self.bm.start_sim_time
#         self.end_sim_time = self.bm.end_sim_time
        
#         self.stumps=ensembler_stumps
#         self.stump_hps = [s.hyperparams.from_default(s.device) for s in self.stumps]
        
#         for m in self.stumps: m.init_sim_matr(1, m.hyperparams.from_default(device=m.device))
# #         self.n_models = len(models)
# #         self.ensembler = ensembler
        
# #         ensembler_dataset = MoE_Dataset(ensembler_stumps)
# #         train_set, val_set = torch.utils.data.random_split(ensembler_dataset, [0.8, 0.2])
# #         train_loader = DataLoader(train_set, batch_size=8192, shuffle=True)
# #         val_loader = DataLoader(val_set, batch_size=8192, shuffle=False)
# #         optimizer = optim.Adam(self.ensembler.parameters(), lr=0.005)

# #         self.trainer = DTrainer(
# #             name, 
# #             self.ensembler,
# #             ensembler_dataset,
# #             train_loader,
# #             val_loader,
# #             None,
# #             moe_a_loss,
# #             optimizer,
# #             None,
# #             "checkpoints/nn_moe",
# #             "results/nn_moe",
# #             save_cycle=10,
# #         )

# #     def load_ensembler(self):
# #         if len(self.trainer.extract_epocks()) == 0:
# #             print("No Ensembler Checkpoint found, fit first!")
# #             return
# #         self.trainer.restore_last_checkpoint()
        
#     def calc_a(self, t, hp):
#         # For each step, simulate each
#         # (nBatch) x (nSeq) x (nModels)
#         preds = torch.stack([mod.calc_a(t, h) for (mod, h) in zip(self.stumps, self.stump_hps)], axis=-1)
#         # Get the conditions, feed it to a weight function
#         # (nBatch)x(nSeq)x(nFeatures)
#         condition = torch.stack([self.v_pred[...,t], self.s_pred[...,t], self.relv_pred[...,t]], axis=-1)
#         cond_flat = condition.reshape(-1, 3).detach().cpu().numpy()
#         # Get the weights as (nBatch)()x(nSeq)x(nModel)
#         weights = self.gmm.predict_proba(cond_flat)
#         weights = torch.from_numpy(weights).reshape(self.mat_shape[0], self.mat_shape[1], 4).to(self.device)
#         # Then produce compounded output, (nBatch)x(nSeq)x(nModel) * (nModel)x1
#         preds = (preds * weights).sum(axis=-1).squeeze()
#         return preds
        
#     def simulate_step(self, t, hp):
#         super().simulate_step(t, hp)
#         for m in self.stumps:
#             m.a_pred[...,t+1] = self.a_pred[...,t]
#             if self.pred_field == PRED_ACC:
#                 m.x_pred[...,t+1] = self.x_pred[...,t+1]
#                 m.s_pred[...,t+1] = self.s_pred[...,t+1]
#                 m.relv_pred[...,t+1] = self.relv_pred[...,t+1]
#             elif self.pred_field == PRED_VEL:
#                 m.x_pred[...,t] = self.x_pred[...,t]
#                 m.s_pred[...,t] = self.s_pred[...,t]
#                 m.relv_pred[...,t] = self.relv_pred[...,t]
        
#         # Synchronize result across all models

# #     def gem_ith(self, base_learners):
        
# #         # Define hyperparameters and models
# #         base_learners = [BaseModel() for _ in range(k)]
# #         hyperparameters = [{'lr': 0.01, 'batch_size': 32}, {'lr': 0.1, 'batch_size': 64}]

# #         # K-fold cross-validation setup
# #         kfold = KFold(n_splits=m, shuffle=True, random_state=42)

# #         # Training and gathering predictions
# #         predictions = []
# #         for train_idx, test_idx in kfold.split(X):
# #             for model in base_learners:
# #                 for params in hyperparameters:
# #                     model.simulate(params)
# #                     predictions.append(modpreds.detach().numpy())  # Storing numpy array of predictions

# #         # Assuming predictions stored as a list of numpy arrays
# #         predictions = np.array(predictions)

# #         def ensemble_loss(weights):
# #             # Weighted sum of predictions
# #             ensemble_pred = np.tensordot(weights, predictions, axes=((0,), (0,)))
# #             mse = np.mean((ensemble_pred - y_test)**2)
# #             return mse

# #         # Initial weights
# #         initial_weights = np.random.rand(len(base_learners))

# #         # Constraint: sum of weights = 1
# #         cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# #         # Bound weights to be between 0 and 1
# #         bounds = [(0, 1) for _ in base_learners]

# #         # Minimize the MSE
# #         result = minimize(ensemble_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=cons)

# #         print('Optimal weights:', result.x)