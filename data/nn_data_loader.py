import warnings
from functools import cached_property

from math import floor

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import torch

from .DTrainer.dtrainer_utils import *


class CongestionData:
  def __init__(self,filename, col_headers=None):
    self.data = loadNPY(filename)
    
    # Add Visual Looming
    W = 2    # Car with as 2 meters by default
    theta = 2 * np.arctan(W, 2*self.data[:,6]).reshape(-1, 1)
    # dtheta = -W*self.data[:,7]/(self.data[:,6]**2 + W**2 / 4).reshape(-1,1)
    
    self.data = np.concatenate([self.data, theta], axis=1)
    print("Added Visual Looming parameters")

    self.col_headers = None
    if col_headers is None:
      warnings.warn("Column Not Defined! Building ID-indexing instead")
    else:
      self.col_headers = col_headers
      print("Retrieved header: ", self.col_headers)

  def _col_num(self, header_name):
    if self.col_headers is None:
      raise KeyError("No header Provided!")
    if header_name not in self.col_headers.keys():
      raise KeyError(f"Invalid Key {header_name}!")
    return self.col_headers[header_name]
  
  """
  Per Driver Dictionary
  """
  @cached_property
  def by_driver_dict(self):
    """Builds a dictionary of per driver driving dataset"""
    table = self.data
    trips = {}
    last_driver_start_row = 0
    last_driver = -1
    # For each row, check if new driver
    for i in range(table.shape[0]):

      # New Driver
      driver = table[i,0].astype(np.uint16)
      if driver != last_driver:
        if last_driver > 0:
          trips[last_driver] = table[last_driver_start_row:i,:]
        last_driver = driver
        last_driver_start_row = i
    trips[table[-1,0].astype(np.uint16)] = table[last_driver_start_row:,:]
    return trips

  # Get all matrices from driver dict
  def iter_driver_dict(self):
    """Iterates through by_driver_dict, return driver (key), and associated matrix"""
    for driver in self.by_driver_dict.keys():
      yield driver, self.by_driver_dict[driver]


  """
  Per Driver & Trip Dictionary
  """
  @cached_property
  def by_driver_trip_dict(self):
    """builds a dictionary of per driver per trip driving dataset"""
    trips = self.by_driver_dict
    new_trips = {}

    for driver, table in self.iter_driver_dict():
      new_trips[driver] = {}
      last_trip_start_row = 0
      last_trip_id = -1

      # For each row, build dict
      for i in range(table.shape[0]):
        trip = table[i,1].astype(np.uint16)

        # New trip
        if trip != last_trip_id:
          if last_trip_id > 0:
            new_trips[driver][last_trip_id] = table[last_trip_start_row:i,:]
          last_trip_id = trip
          last_trip_start_row = i
      new_trips[driver][table[-1,1].astype(np.uint16)] = table[last_trip_start_row:,:]
    return new_trips
  
  # Get all matrices from driver dict
  def iter_driver_trip_dict(self):
    """Iterates through by_driver_dict, return driver (key), and associated matrix"""
    for driver in self.by_driver_trip_dict.keys():
      for trip in self.by_driver_trip_dict[driver].keys():
        yield driver, trip,self.by_driver_trip_dict[driver][trip]

  """
  Per Driver & Trip & Congestion Dictionary
  """
  @cached_property
  def by_driver_trip_congestion_dict(self):
    """builds a dictionary of per driver per trip driving dataset"""
    trips = self.by_driver_dict
    new_trips = {}

    for driver, trip, table in self.iter_driver_trip_dict():
      if driver not in new_trips.keys(): new_trips[driver] = {}
      if trip not in new_trips[driver].keys(): new_trips[driver][trip] = []
      last_cong_start_row = 0
      last_cong_id = 0

      # For each row, build dict
      for i in range(1,table.shape[0]):
        # If time mismatch, new congestion
        if table[i,2] != table[i-1,2] + 10:
          new_trips[driver][trip].append(table[last_cong_start_row:i,:])
          last_cong_start_row = i
          
      new_trips[driver][trip].append(table[last_cong_start_row:,:])
    return new_trips
  
  # Get all matrices from driver dict
  
  def iter_driver_trip_congestion_dict(self):
    """Iterates through by_driver_dict, return driver (key), and associated matrix"""
    for driver in self.by_driver_trip_dict.keys():
      for trip in self.by_driver_trip_dict[driver].keys():
        i = 0
        for i, congestion in enumerate(self.by_driver_trip_congestion_dict[driver][trip]):
          yield driver, trip, i, self.by_driver_trip_congestion_dict[driver][trip][i]



  @property
  def longest_trip_len(self):
    return max(table.shape[0] for _, table in self.iter_driver_dict())

  def filter2dict(self, filterer):
    new_trips = {}
    for driver, trip, cid, table in self.iter_driver_trip_congestion_dict():
      # Init dict
      if driver not in new_trips.keys(): new_trips[driver] = {}
      if trip not in new_trips[driver].keys(): new_trips[driver][trip] = []

      # Initialize search
      last_idx = 0
      recording = True
      for i in range(1,table.shape[0]):
        if filterer.should_start(table, i):
          last_idx = i
          recording = True
        if filterer.should_end(table, i):
          new_trips[driver][trip].append(table[last_idx:i,:])
          recording = False
      if recording == True:
        new_trips[driver][trip].append(table[last_idx:,:])
    return new_trips

  @cached_property
  def accel_sequence_dict(self):
    acc_filterer = AccelFilter()
    return self.filter2dict(acc_filterer)

  @cached_property
  def moving_sequence_dict(self):
    mov_filterer = MovingFilter()
    return self.filter2dict(mov_filterer)

  @cached_property
  def decel_sequence_dict(self):
    dec_filterer = DecelFilter()
    return self.filter2dict(dec_filterer)


  """
  Properties
  """
  def replicate_struct(self, data):
    """Recreate data structure up until the element level"""
    rtn = None
    if isinstance(data, dict):
      rtn = {}
      for key in data.keys():
        rtn[key] = self.replicate_struct(data[key])
      return rtn
    elif isinstance(data, list):
      rtn = []
      for item in data:
        st = self.replicate_struct(item)
        if st is not None:
          rtn.append(st)
      return rtn
    return None

  def len(self, data=None, count_table_row=False):
    """Get number of data in composite structure"""
    if data is None: 
      return self.data.shape[0]
    total_len = 0
    if isinstance(data, dict):
      for key in data.keys():
        total_len += self.len(data[key])
      return total_len
    elif isinstance(data, list):
      for item in data:
        total_len += self.len(item)
      return total_len
    elif isinstance(data, np.ndarray) and count_table_row: return data.shape[0]
    return 1

  def len_dict(self, data, keepdims=-1):
    """Build number of data composite structure"""
    if keepdims==-1:
      return self.len(data, count_table_row=True)
    elif isinstance(data, dict):
      new_data = self.replicate_struct(data)
      for key in data.keys():
        new_data[key] = self.len_dict(data[key], keepdims-1)
      return new_data
    elif isinstance(data, list):
      new_data = self.replicate_struct(data)
      for item in data:
        new_data.append(self.len_dict(item, keepdims-1))
      return new_data
    elif isinstance(data, np.ndarray):
      return data.shape[0]
    return 0
  
  def visualize_dict(self, data, headers, depth=0, depth_limit=-1, max_num=-1):
    """
    Build recursive tree to visualize data in dictionary / list composition
    """
    num_printed = 0
    if depth==depth_limit:
      print('|   '*depth + headers[depth] + ": " + str(self.len(data)) + "x items")
      return

    if isinstance(data, dict):
      print('|   '*depth + headers[depth] + ": " + str(len(data.keys())) + "x items")
      for key in data.keys():
        print('|   '*depth + '|- ' + headers[depth] + " " + str(key))
        self.visualize_dict(data[key], headers, depth+1, depth_limit, max_num)
        num_printed += 1
        if num_printed >= max_num and max_num > 0: break
      return
    if isinstance(data, list):
      print('|   '*depth + headers[depth] + ": " + str(len(data)) + "x items")
      for table in data:
        print('|   '*depth + '|- item ' + headers[depth] + " " + str(num_printed))
        self.visualize_dict(table, headers, depth+1, depth_limit, max_num)
        num_printed += 1
        if num_printed >= max_num and max_num > 0: break
      return
    if isinstance(data, np.ndarray):
      print('|   '*depth + '|- NumPy array with shape ' + str(data.shape))
      return
    print('|   '*depth + '|- Unknown data type: ' + str(data))

  def dict2table(self, data):
    table = None
    if isinstance(data, dict):
      for key in data.keys():
        if table is None: table = self.dict2table(data[key])
        else: 
          table = np.concatenate((table, self.dict2table(data[key])),axis=0)
      return table
    if isinstance(data, list):
      for item in data:
        if table is None: table = self.dict2table(item)
        else: 
          table = np.concatenate((table, self.dict2table(item)),axis=0)
      return table
    if isinstance(data, np.ndarray):
      return data
    if isinstance(data, int):
      return np.array([data])
    raise Exception(f"Unhandlable data type {type(data)} encountered!")

  @property
  def n_driver(self):
    return self.len(self.by_driver_dict)

  @property
  def n_trips(self):
    return self.len(self.by_driver_trip_dict)

  @property
  def n_congestion(self):
    return self.len(self.by_driver_trip_congestion_dict)

  @property
  def n_accel_seq(self):
    return self.len(self.accel_sequence_dict)

  @property
  def n_moving_seq(self):
    return self.len(self.moving_sequence_dict)

  @property
  def n_decel_seq(self):
    return self.len(self.decel_sequence_dict)

  @property
  def n_trips_per_driver(self):
    lens = {}
    trips = self.by_driver_trip_dict
    for driver in trips.keys():
      lens[driver] = len(trips[driver].keys())
    return lens

  @property
  def len_per_drive_trip(self):
    lens = {}
    trips = self.by_driver_trip_dict
    for driver in trips.keys():
      lens[driver] = {}
      for trip in trips[driver].keys():
        lens[driver][trip] = trips[driver][trip].shape
    return lens
  
  @property
  def __len__(self):
    return self.data.shape[0]

  @staticmethod
  def feature_label_split(table, feature_cols=np.array([3,4,6,7,8,9,12,13,14]), label_col=5):
    return table[:, feature_cols], table[:,label_col]


class AccelFilter:
  def __init__(self):
    pass

  def should_start(self, table, i):
    return table[i,4] > 0.1 and table[i-1,4] < 0.1

  def should_end(self, table, i):
    return table[i,4] < 0.1 and table[i-1,4] > 0.1


class DecelFilter:
  def __init__(self):
    pass  

  def should_start(self, table, i):
    return table[i,14] > 0.5 and table[i-1,14] < 0.5

  def should_end(self, table, i):
    return table[i,14] < 0.5 and table[i-1,14] > 0.5


class MovingFilter:
  def __init__(self):
    pass  

  def should_start(self, table, i):
    return table[i,3] > 0.1 and table[i-1,3] < 0.1

  def should_end(self, table, i):
    return table[i,3] < 0.1 and table[i-1,3] > 0.1


class JamNoAcc(Dataset):
  """
  Dataset for JamDetailWBT.npy
  """
  def __init__(self, len_sequence=1, average_dist=0, label_cnt=1, feature_cols=[3,6,7,8,9,12,13,14], label_col=5):
    """
    len_sequence: length of each sequence feature
    average_dist: average distance between sampled starting points
    label_cnt: take the last how many accelerations as label
    """
    source = CongestionData("raw_data/JamDetailWBT.npy")
    
    # Set random seed
    np.random.seed(42)
    data_list = []
    label_list = []

    # Sample sequences
    for driver, trip, congest, table in source.iter_driver_trip_congestion_dict():
      if table.shape[0] < len_sequence: continue
      if average_dist < 1:
        indices = np.arange(table.shape[0])
      else: 
        indices = np.random.randint(low=0, high=table.shape[0]-len_sequence, \
                                    size=table.shape[0]//average_dist)
      for idx in indices:
        block = table[idx:idx+len_sequence,:]
        feature, label = CongestionData.feature_label_split(block, feature_cols, label_col)

        # Add to data list
        data_list.append(feature.flatten())
        label_list.append(label[-1])

    # Shuffle together, then 
    data = np.array(data_list)
    labels = np.array(label_list)
    data = np.concatenate((data, labels[:,None]),axis=1)
    np.random.seed(42)
    data = np.random.permutation(data)

    # Restore data and labels
    self.features = torch.from_numpy(data[:,:-1]).float()
    self.labels = torch.from_numpy(data[:,-1]).float()

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.to_list()
    return self.features[idx,:], self.labels[idx, None]
    
  def __len__(self):
    return self.features.shape[0]

  def num_features(self):
    return self.features.shape[1]

  def num_labels(self):
    return self.labels.shape[1]


# INCORRECT IMPLEMENTATION FOR LSTM
# class Jam2Dataset(Dataset):
#   """
#   Dataset for JamDetailWBT.npy, where the input is previous len_sequence features, and labels is acceleration of next len_label accelerations.
#   """
#   def __init__(self, len_sequence=1, average_dist=0, len_label=1, feature_cols=[3,5,6,7,8,9,12,13,14], label_col=5):
#     """
#     len_sequence: length of each sequence feature
#     average_dist: average distance between sampled starting points
#     label_cnt: take the last how many accelerations as label
#     """
#     source = CongestionData("data/raw_data/JamDetailWBT.npy")
    
#     # Set random seed
#     np.random.seed(42)
#     data_list = []
#     label_list = []

#     # Sample sequences
#     for driver, trip, congest, table in source.iter_driver_trip_congestion_dict():
#       if table.shape[0] < len_sequence+len_label: continue
#       if average_dist < 1:
#         indices = np.arange(table.shape[0]-len_sequence-len_label)
#       else: 
#         indices = np.random.randint(low=0, high=table.shape[0]-len_sequence-len_label, \
#                                     size=table.shape[0]//average_dist)
#       for idx in indices:
#         feature = table[idx:idx+len_sequence,feature_cols]
#         label = table[idx+len_sequence:idx+len_sequence+len_label,label_col]
#         # feature, label = CongestionData.feature_label_split(block, feature_cols, label_col)

#         # Add to data list
#         data_list.append(feature.flatten())
#         label_list.append(label)
    
#     # Restore data and labels
#     self.features = torch.from_numpy(np.array(data_list)).float()
#     self.labels = torch.from_numpy(np.array(label_list)).float()

#   def __getitem__(self, idx):
#     if torch.is_tensor(idx):
#       idx = idx.to_list()
#     return self.features[idx,:], self.labels[idx,:]
    
#   def __len__(self):
#     return self.features.shape[0]

#   def num_features(self):
#     return self.features.shape[1]

#   def num_labels(self):
#     return self.labels.shape[1]


class Jam2DDataset(Dataset):
    """
    Dataset for JamDetailWBT.npy, where the input is previous len_sequence features, and labels is acceleration of next len_label accelerations.
    """
    def __init__(self, len_sequence=1, average_dist=0, len_label=1, feature_cols=[3,5,6,7,8,9,12,13,14], label_col=5, sampling="random"):
        """
        len_sequence: length of each sequence feature
        average_dist: average distance between sampled starting points
        label_cnt: take the last how many accelerations as label
        """
        source = CongestionData("data/JamDetailWBT.npy")
        
        print("Source loaded")
    
        # Set random seed
        np.random.seed(42)
        data_list = []
        label_list = []

        # Sample sequences
        for driver, trip, congest, table in source.iter_driver_trip_congestion_dict():
            if table.shape[0] < len_sequence+len_label: continue
            if average_dist < 1:
                indices = np.arange(table.shape[0]-len_sequence-len_label)
            elif sampling == "random":
                indices = np.random.randint(low=0, high=table.shape[0]-len_sequence-len_label, \
                                        size=table.shape[0]//average_dist)
            elif sampling == "sequential":
                indices = np.arange(0, table.shape[0]-len_sequence-len_label, average_dist)
            else:
                raise Exception("Invalid sampling method!")
            for idx in indices:
                feature = table[idx:idx+len_sequence,feature_cols]
                label = table[idx+len_sequence:idx+len_sequence+len_label,label_col]
                # feature, label = CongestionData.feature_label_split(block, feature_cols, label_col)

                # Add to data list
                data_list.append(feature)
                label_list.append(label)

        # Restore data and labels
        self.features = torch.from_numpy(np.array(data_list)).float()
        print(self.features.size())
        self.labels = torch.from_numpy(np.array(label_list)).float()
        print(self.labels.size())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        return self.features[idx,:], self.labels[idx,:]

    def __len__(self):
        return self.features.shape[0]

    def num_features(self):
        return self.features.shape[-1]

    def num_labels(self):
        return self.labels.shape[-1]


    
    
    
    
    

      
# class Jam2DPDDataset(Dataset):
#     """
#     Dataset for JamDetailWBT.npy, where the input is previous len_sequence features, and labels is acceleration of next len_label accelerations.
#     """
#     def __init__(self, source, feature_cols, label_cols, len_sequence=1, sampling='random', average_dist=1):
#         """
#         len_sequence: length of each sequence feature
#         average_dist: average distance between sampled starting points
#         label_cnt: take the last how many accelerations as label
#         """

#         data_list = []
#         label_list = []

#         # Sample sequences
#         for key, table in source:
#             if table.shape[0] < len_sequence+1: 
#                 continue
#             else: 
#                 if sampling == 'sequential':
#                     indices = np.arange(0, table.shape[0]-len_sequence-1, average_dist)
#                 elif sampling == 'random':
#                     np.random.seed(42)
#                     indices = np.random.randint(low=0, high=table.shape[0]-len_sequence-1, \
#                                         size=table.shape[0]//average_dist)
#                 else:
#                     raise Exception("Unsupported sampling method!")
                    
#             for idx in indices:
#                 feature = table[idx:idx+len_sequence,feature_cols]
#                 label = table[idx+len_sequence:idx+len_sequence+len_label,label_col]
#                 # feature, label = CongestionData.feature_label_split(block, feature_cols, label_col)

#                 # Add to data list
#                 data_list.append(feature)
#                 label_list.append(label)

#         # Restore data and labels
#         self.features = torch.from_numpy(np.array(data_list)).float()
#         print(self.features.size())
#         self.labels = torch.from_numpy(np.array(label_list)).float()
#         print(self.labels.size())

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.to_list()
#         return self.features[idx,:], self.labels[idx,:]

#     def __len__(self):
#         return self.features.shape[0]

#     def num_features(self):
#         return self.features.shape[-1]

#     def num_labels(self):
#         return self.labels.shape[-1]
      
# #     def visualize_overlap(self):
        
        
        
# class Jam2DPDManager():
#     """Splitting CongestionPD for train, val, test sets"""
#     def __init__(self, feature_cols, label_cols, len_sequence=1, sampling='random', average_dist=1):
#         source = CongestionPD("JamDetailWBreak.csv")
#         total_len = 0
#         seq_lens = []
#         seqs = [] 
#         for key, table in source:
#             total_len += table.shape[0]
#             seq_lens.append(table.shape)
#             seqs.append(table)
            
#         randomized_idx = np.random.permutation(len(seq_lens))
#         idx, train_set_len, train_set_raw = self.add2set(randomized_idx, 0, total_len*0.7)
#         idx, val_set_len, val_set_raw = self.add2set(randomized_idx, idx, total_len*0.2)
#         idx, test_set_len, test_set_raw = self.add2set(randomized_idx, idx, total_len*0.2)
        
#         print(f"Acquired train set with aggregated sequence length {train_set_len}")
#         print(f"Acquired validation set with aggregated sequence length {val_set_len}")
#         print(f"Acquired test set with aggregated sequence length {test_set_len}")
        
#         self.train_set = Jam2DPDDataset(train_set)
#         self.val_set = Jam2DPDDataset(val_set)
#         self.test_set = Jam2DPDDataset(test_set)
        
    
#     def add2set(self, rand_seq, seqs, idx, max_len):
#         candidate_set = []
#         idx = 0
#         set_len = 0
        
#         while (set_len < max_len) and idx < len(rand_seq):
#             candidate_set.append(seqs[idx])
#             set_len = seqs[idx].shape[0]
#             idx += 1
            
#         return idx+1, set_len, candidate_set
        