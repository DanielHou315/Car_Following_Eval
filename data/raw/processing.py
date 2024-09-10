from utils import *
import numpy as np

config = {
  "load_clean":True,
  "add_brake":True,
  "add_t_since_congest":True,
  "add_t_since_sequence":True,
  "save_table":False,
}

"""
Load csv file, clean 
"""
def load_clean_csv(filename):
  return cleanTable(loadCSV(filename))


""" 
Add Brake Binary Data from CandiTrips
"""
def add_brake_binary():
  # Get candidate trip data dict
  CandiTrips = CongestionData("data/CandiTrip.npy")
  candi_dict = CandiTrips.by_driver_trip_dict_cols(cols=(4,5))

  jams = loadNPY("data/JamDetail.npy")
  brake_event = np.zeros(shape=jams.shape[0],dtype=jams.dtype)

  # For each row, build dict
  for i in range(jams.shape[0]):
    driver = jams[i,0].astype(np.uint16)
    trip = jams[i,1].astype(np.uint16)

    ranges = candi_dict[driver][trip]

    for j in range(ranges.shape[0]):
      # If reached larger start, no break
      if ranges[j,0] > jams[i,2]:
        break
      # if end too small, keep going
      if ranges[j,1] < jams[i,2]:
        continue
      # If in between, yes and break
      if ranges[j,0] <= jams[i,2] and ranges[j,1] >= jams[i,2]:
        brake_event[i] = 1.0
        break
  return np.concatenate((jams, brake_event[:,None]), axis=1)

""" 
Append Time Since Congestion Info
"""
def append_time_info():
  printed = 0
  trips = CongestionData("data/JamDetailWBreak.npy")
  running_row_cnt = 0
  new_data = None
  for driver, trip, table in trips.iter_driver_trip_dict():
    last_start = 0
    dtime = np.zeros(shape=table.shape[0],dtype=table.dtype)
    start_row = running_row_cnt * np.ones(shape=table.shape[0],dtype=table.dtype)
    for i in range(1,table.shape[0]):
      if table[i,2] != table[i-1,2] + 10:
        dtime[last_start:i] = np.arange(0, (i-last_start) * 10, 10)
        start_row[last_start:i] = running_row_cnt * np.ones(shape=(i-last_start),dtype=table.dtype)
        running_row_cnt = running_row_cnt + i - last_start
        
        if printed < 5:
          print(f"Found change in driver {driver} trip {trip} at {i} where last time is {last_start}")
          print("Setting dtime", dtime[last_start:i])
          printed += 1
        last_start = i
    dtime[last_start:] = np.arange(0, (table.shape[0]-last_start) * 10, 10)
    start_row[last_start:] = running_row_cnt * np.ones(shape=(table.shape[0]-last_start),dtype=table.dtype)
    running_row_cnt = running_row_cnt + table.shape[0] - last_start
    new_table = np.concatenate((table, dtime[:,None], start_row[:,None]), axis=1)
    if new_data is None:
      new_data = new_table
    else:
      new_data = np.concatenate((new_data, new_table), axis=0)
  return new_data


def main():

  jams = CongestionData("data/JamDetailWBT.npy")
  print(jams.longest_trip_len)

  # Load and clean option
  if config["load_clean"]:
    tables = []
    for filename in ["data/CandiTrip.csv", "data/JamDetail.csv", "data/JamEvents.csv"]:
      table = load_clean_csv(filename)
      tables.append(table)
      if config["save_table"]: saveTable(table, filename[:,-4])
  
  # Add Brake Option
  if config["add_brake"]:
    table = add_brake_binary()
    if config["save_table"]:
      saveTable(table, "data/JamDetailWBreak")
      np.savetxt("data/JamDetailWBreak.csv",table,delimiter=",",fmt="%.7f")
    
  # Add Time Since Congestion
  if config["add_t_since_congest"]:
    table = append_time_info()
    if config["save_table"]:
      saveTable(table, "data/JamDetailWBT")
      np.savetxt("data/JamDetailWBT.csv",table,delimiter=",",fmt="%.7f")

  # Add Time Since Congestion
  if config["add_t_since_congest"]:
    table = append_time_info()
    if config["save_table"]:
      saveTable(table, "data/JamDetailWBT")
      np.savetxt("data/JamDetailWBT.csv",table,delimiter=",",fmt="%.7f")


if __name__ == "__main__":
  main()