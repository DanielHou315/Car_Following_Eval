from utils import *

# def plot_accel_hist()

def exp_acc_vs_range(data_loader):
  plot_acc_dist(data_loader, os.path.join(os.path.dirname(os.path.realpath(__file__)), "graphs/acc_vs_range2"))


def tree_dicts(data_loader, query):

  # Visualize first three separations
  if query == "driver_dict": data_loader.visualize_dict(data_loader.by_driver_dict, ["driver"])
  elif query == "driver_trip_dict": data_loader.visualize_dict(data_loader.by_driver_trip_dict, ["driver", "trip"])
  elif query == "driver_trip_congestion_dict":  data_loader.visualize_dict(data_loader.by_driver_trip_congestion_dict, ["driver", "trip", "congestion ID"])

  # This is too much, so we instead summarize
  elif query == "accel_summary": data_loader.visualize_dict(data_loader.accel_sequence_dict, ["driver", "trip", "accelID", "Item", "Item"], depth=0, depth_limit=2,max_num=5)
  
  # We can summarize the data and print in tree manner
  # data_loader.visualize_dict(data_loader.len_dict(data_loader.accel_sequence_dict, keepdims=2), ["driver", "trip", "accelID", "Item", "Item"], depth=0, depth_limit=-1,max_num=-1)
  elif query == "decel_summary": data_loader.visualize_dict(data_loader.decel_sequence_dict, ["driver", "trip", "decelID", "Item", "Item"], depth=0, depth_limit=2,max_num=-1)

  # Moving sequences likewise
  elif query == "moving_summary": data_loader.visualize_dict(data_loader.moving_sequence_dict, ["driver", "trip", "moveID", "Item", "Item"], depth=0, depth_limit=2,max_num=-1)

  else:
    print("Invalid query argument!")


def plot_hist_adm(data):
  """
  plot histogram of distribution of accel, decel, moving number of events per trip
  data: list of n arrays
  """
  fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(60,20))
  plot_hist(ax1, data[0], "N acc per trip", nbins=100)
  plot_hist(ax2, data[1], "N dec per trip", nbins=100)
  plot_hist(ax3, data[2], "N mov per trip", nbins=100)
  plt.savefig("graphs/summary/hist_adm.png")
  plt.close()


def plot_acc_dec_mov_distribution(data_loader):
  """data_loader: data loader object"""

  print("Number of trips:", data_loader.n_trips)

  # print(data_loader.len_dict(data_loader.accel_sequence_dict, keepdims=2))
  # summarize num of accel events to per trip level
  accel_sum = data_loader.dict2table(data_loader.len_dict(data_loader.accel_sequence_dict, keepdims=1))
  print(accel_sum.max()) # summarize to per trip level

  data_loader.visualize_dict(data_loader.len_dict(data_loader.accel_sequence_dict, keepdims=1), headers=["driver", "trip", "accID", "Item", "Item"])

  # summarize num of decel to per trip level
  decel_sum = data_loader.dict2table(data_loader.len_dict(data_loader.decel_sequence_dict, keepdims=1))
  print(decel_sum.max()) # summarize to per trip level

  # summarize num of decel to per trip level
  moving_sum = data_loader.dict2table(data_loader.len_dict(data_loader.moving_sequence_dict, keepdims=1))
  print(moving_sum.max()) # summarize to per trip level

  plot_hist_adm([accel_sum, decel_sum, moving_sum])
  

def main():
  data_loader = CongestionData("data/JamDetailWBT.npy")
  # ree_dicts(data_loader, query="moving_summary")
  plot_acc_dec_mov_distribution(data_loader)

if __name__ == "__main__":
  main()