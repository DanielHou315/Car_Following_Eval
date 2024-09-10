import numpy as np
import matplotlib.pyplot as plt


def main():
  """
  driver to run spectral clustering
  """
  # Load data
  jam_detail = np.genfromtxt('../data/JamDetail.csv', delimiter=',')
  print("Found data with shape", jam_detail.shape)

  # Select data for clustering
  rows_of_interest = [0,1,2,5,6]
  jam_intr = jam_detail[:,rows_of_interest]
  # Remove nan
  jam_intr = jam_intr[~np.isnan(jam_intr).any(axis=1)]

  # Extract for each driver
  jam_driver = {}
  # Driver ID
  for i in range(1, np.max(jam_intr[:,0]).astype(np.uint16)):
    driver_data = None
    # Trip ID
    for j in range(1, np.max(jam_intr[:,1]).astype(np.uint16)):
      # Filtering: valid driver
      # Filtering: valid trip
      
      mask = np.logical_and(jam_intr[:,0] == i, jam_intr[:,1] == j)
      # Filtering: > 0 range to remove invalid measurements
      mask = np.logical_and(mask, jam_intr[:,4] > 0)
      # Filtering: != 0 acceleration to remove static ranges
      mask = np.logical_and(mask, jam_intr[:,3] != 0)
      # Acquire filtered data
      data = jam_intr[mask,2:]
      # Check for empty driver
      if data.size < 1: continue

      # Otherwise, deal with time and make it delta (time since congestion event)
      data[:,0] = data[:,0] - data[0,0]
      
      # Concat data
      if driver_data is None: driver_data = data
      else: driver_data = np.concatenate((driver_data, data), axis=0)

    # Check for empty driver
    if driver_data is None: continue
    # Otherwise, put data in and 
    jam_driver[i] = driver_data
    plotAccDist(jam_driver[i], i)

if __name__ == "__main__":
  main()