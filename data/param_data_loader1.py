import numpy as np
import pandas as pd


SAVE2CSV = True

class AutoIncrementer:
    def __init__(self, init_val=0, increment=1):
        self.val = init_val
        self.increment = increment

    def __call__(self):
        self.val += self.increment
        return self.val - self.increment

class CongestionPD:
    def __init__(self):
        filename = "data/JamDetailWBreak.csv"
        if not filename.endswith('csv'):
            raise Exception("Data file MUST BE CSV!")
        self.df = pd.read_csv(filename)
        self.tick = 0.1

        self.congestions = self.separate_by_congestion(self.df)
        print(f"Separated {len(self.congestions)} congestion events")
        self.cf_events = self.separate_by_cf(self.congestions)
        self.cf_events = self.filter_by_length(self.cf_events)
        print(f"Separated {len(self.cf_events)} car following events")

        # Add more fields, by congestion
        for i, (key, df) in enumerate(self.cf_events.items()):
            
            next_col = AutoIncrementer(df.shape[1])
            # time since congestion
            df.insert(next_col(), "timeSinceCongestion", df.loc[:, "time"]-df.loc[df.index[0],"time"])

            # Add Visual Looming Information
            W = 2    # Car with as 2 meters by default
            df.insert(next_col(), "VL_theta", self.calc_vl_theta(df, W))
            df.insert(next_col(), "VL_dtheta", self.calc_vl_dtheta(df, W))

            # Add computed lead vehicle information
            df.insert(next_col(), "lv_speed", self.calc_lv_speed(df))
            df.insert(next_col(), "lv_Ax", self.calc_lv_acc(df))

            if SAVE2CSV:
                df.to_csv(f"data/pd_cached_cf/congestion_{i}_processed.csv")

        print("Datast Loaded. Preview of dataset: ")
        print(list(self.cf_events.values())[0].head())

    def calc_vl_theta(self, df, W) -> np.ndarray:
        return 2 * np.arctan2(W, 2*df.loc[:, "range"])
                            
    def calc_vl_dtheta(self, df, W) -> np.ndarray:
        return -W * df.loc[:,"rangerate"] / (df.loc[:,'range']**2 + W**2 / 4)
    
    @staticmethod
    def calc_lv_speed(df) -> pd.Series:
        return df.loc[:, "speed"] + df.loc[:, "rangerate"]

    def calc_lv_acc(self, df) -> pd.Series:
        lv_v_diff = np.diff(df.loc[:,"lv_speed"].to_numpy())
        acc = lv_v_diff / self.tick
        acc = np.insert(acc, 0, acc[0])
        return pd.DataFrame(acc, index=df.index)
    
    def calc_lv_x(self):
        """calculates distance travelled by lead vehicle since start of ticking"""
        raise NotImplementedError("Requires query of position and headway to implement")

    def separate_by_congestion(self, table: pd.DataFrame) -> dict:
        """
        Returns dictionary with key (driver, trip id, congestion id) and value type pd.dataframe being that segment
        """
        start_row = 0
        next_congest_id = 0
        trip_dict ={}
        
        for i in range(table.shape[0]):
            # New congestion event
            if (i == table.shape[0] - 1) or self.is_trip_end(table, i):
                key = (table.loc[i, 'driver'], table.loc[i, 'trip'], next_congest_id)
                val = table.loc[start_row:i]
                
                # Add congestion ID to data table
                val.insert(2, "congestion", np.ones(shape=(val.shape[0],), dtype=np.uint16) * next_congest_id)
                
                # Add to dictionary
                trip_dict[key] = val
                next_congest_id += 1
                start_row = i+1
        return trip_dict
    
    def separate_by_cf(self, congestions):
        """Filter out segments where """
        cf_events = {}
        for i, (key, df) in enumerate(congestions.items()):
            # Create a boolean series where the condition is met
            condition = df['range'] != 0
            changes = condition.astype(int).diff().fillna(0).abs()
            groups = changes.cumsum()
            dfs = [group_df for _, group_df in df.groupby(groups)]
            for i, df in enumerate(dfs):
                cf_events[key + (i,)] = df
        return cf_events
    
    def filter_by_length(self, cf_events: dict, min_len: int =50) -> dict:
        """events less than 5 seconds are ignored"""
        keys2del = []
        for key, val in cf_events.items():
            if len(val) < min_len:
                keys2del.append(key)
        for key in keys2del:
            del cf_events[key]
        return cf_events

    def __iter__(self):
        """iterator"""
        # for key, val in self.congestions.items():
        for key, val in self.cf_events.items():
            yield key, val
        
    # Helper Functions
    def is_trip_end(self, table: pd.DataFrame, i: int) -> bool:
        return ((table.loc[i, 'driver'] != table.loc[i+1, 'driver'])   # Different Driver
            or (table.loc[i, 'trip'] != table.loc[i+1, 'trip'])       # Different Trip
            or (table.loc[i+1, 'time'] - table.loc[i, 'time'] != 10)) # Different timestamps