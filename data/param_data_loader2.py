import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn.mixture

# from torch import P
from tslearn.clustering import TimeSeriesKMeans

pd.options.mode.chained_assignment = None  # default='warn'

# from sklearn.metrics import root_mean_squared_error as rmse

class JamSeq:
    def __init__(self, key_dict, df):
        self.df = df
        self.key_dict = key_dict
        for k, val in key_dict:
            setattr(self, k, val)

class AutoIncrementer:
    def __init__(self, init_val=0, increment=1):
        self.val = init_val
        self.init_val = init_val
        self.increment = increment

    def __call__(self):
        self.val += self.increment
        return self.val - self.increment
    
    # def reset(self):
    #     self.val = self.init_val


class ParamDL2:
    TICK = 0.1
    cong_idx_generator = AutoIncrementer()
    cf_idx_generator = AutoIncrementer()

    TYPE_JAM = 0
    TYPE_PRE = 1
    TYPE_POST = 2
    
    CONG_CSV = "data/param_dataset2_cong_highway.csv"
    CONG_PKL = "data/param_dataset2_cong_highway.pkl"
    CF_CSV = "data/param_dataset2_cf_highway.csv"
    CF_PKL = "data/param_dataset2_cf_highway.pkl"

    def __init__(self, build_distribution=False, exclude_cf=None, test_cf=None):
        
        self.exclude_cf = exclude_cf
        self.test_cf = test_cf
        
        # if not os.path.isfile(self.CONG_CSV) or build_distribution is True:
        # Split into congestions, pre-congestions, post-congestions
        print("Processing Congestion")
        self.congs = self.process_congestion()
        # self.congs = self.load_cong()
        self.congests = [df for df in self.__congests]
        # self.pre_congests = [df for df in self.__pre_congests]
        # self.post_congests = [df for df in self.__post_congests]

        # if not os.path.isfile(self.CF_CSV) or build_distribution is True:
        # Further split into car following events
        print("Processing Car Following")
        self.cf_events = self.process_cf(self.congs, min_len=100)
        # self.cf_events = self.load_cf()
        self.cfs = [df for df in self.__cfs]
        # self.pre_cfs = [df for df in self.__pre_cfs]
        # self.post_cfs = [df for df in self.__post_cfs]

        print("Datast Loaded. Preview of dataset: ")
        print(self.cf_events[0].head())

    # -------------------------------
    #
    # PROCESS BY CONGESTION
    # 
    # -------------------------------
    def process_congestion(self):
        files = ["data/param_dataset2.csv"]
                 # , "data/param_dataset2_before.csv"
                 # , "data/param_dataset2_after.csv"]
        jam_dfs = [pd.read_csv(file) for file in files]
        self.master = pd.concat([df for df in jam_dfs])
        
        # Build GMM Model
        self.n_cluster = 4
        self.gmm = sklearn.mixture.GaussianMixture(self.n_cluster, init_params='k-means++')
        gmm_X = pd.concat([df[["speed", "range", "rangerate"]] for df in jam_dfs]).to_numpy()
        gmm_y = pd.concat([df["Ax"] for df in jam_dfs]).to_numpy()
        self.gmm.fit(gmm_X, gmm_y)
        proba = self.gmm.predict_proba(gmm_X)
        sorted_idx = np.argsort(proba, axis=1)
        for i, df in enumerate(jam_dfs):
            df.insert(3, "gmm_n_cluster", self.n_cluster)
            for r in range(sorted_idx.shape[1]):
                df.insert(4+r, f"rank_{r}_cluster", sorted_idx[:, r])
                
        congs = [self.split_by_congestion(df, i) for i,df in enumerate(jam_dfs)]
        jams = congs[0]
        
        highway_seqs = self.highway_jam_seqs(jams)
        jams = [jams[i] for i in highway_seqs]
        self.n_jams = len(jams)
        
        # jams.extend([congs[1][i] for i in highway_seqs])
        # jams.extend([congs[2][i] for i in highway_seqs])
        print(f"Processed {len(jams)} Congestions")

        concat_df = pd.concat(jams)
        concat_df.to_csv(self.CONG_CSV)
        concat_df.to_pickle(self.CONG_PKL)
        print("Saved Congestion File")
        return jams
    
    def highway_jam_seqs(self, jams):
        brake_events = pd.read_csv("data/raw/Task789_AllBrakeEvents.csv")
        seqs = []
        for i, df in enumerate(jams):
            brakes_during_jam = brake_events[
                (brake_events["Driver"] == df.loc[df.index[0], "driver"])
                & (brake_events["Trip"] == df.loc[df.index[0], "trip"])
                & ((brake_events["BrakeStart"]/100 - 0.1) >= df.loc[df.index[0], "time"])
                & ((brake_events["BrakeEnd"]/100 - 0.1) <= df.loc[df.index[-1], "time"])
            ]
            ratio = (brakes_during_jam["Roadtype"]==1).sum() / len(brakes_during_jam)
            
            # If >= half is 1, add as sequence
            if ratio > 0.1:
                seqs.append(i)
        return seqs
            

    @classmethod
    def split_by_congestion(cls, df: pd.DataFrame, tp) -> list[pd.DataFrame]:
        """
        Returns a list of tuples with key (dict) and value type pd.DataFrame being that segment.
        """
        # cls.cong_idx_generator.reset()
        # Type: 0 for regular, 1 for pre, 2 for post
        trip_end = (
            (df['driver'] != df['driver'].shift(1)) |
            (df['trip'] != df['trip'].shift(1)) |
            (df['time'] - df['time'].shift(1) != 10)
        ).astype(int)
        df.insert(2, "type", tp)
        df["cong_idx"] = trip_end.cumsum()
        
        trips = []
        for _, group in df.groupby('cong_idx'):
            if len(group) < 1:
                continue
            group = group.copy()  # Avoid SettingWithCopyWarning
            group.insert(2, "congestion", cls.cong_idx_generator())
            group = group.drop("cong_idx",axis='columns')
            trips.append(cls.compute_cong_features(group))
        return trips
            
    @staticmethod
    def same_feature(df1, df2, field):
        return df1.loc[df1.index[0],field] == df2.loc[df2.index[0], field]
    
    # -------------------------------
    #
    # PROCESS BY CF
    # 
    # -------------------------------
    def process_cf(self, jams, min_len=50):
        # Split into car following events
        cf_events = []
        if self.test_cf is not None: test_events = []
        for df in jams:
            df['range_lost'] = ((df["range"]<0.5) & (df["range"].shift(1)>0.5)).astype(int)
            df['target_change'] = ((df["targetid"] != df["targetid"].shift(1))).astype(int)
            df["cf_group"] = (df["range_lost"] + df["target_change"]).cumsum()

            for _, group in df.groupby('cf_group'):
                group = group[group["range"] > 0.5].copy()
                if len(group) < min_len: continue
                cf_idx = self.cf_idx_generator()
                group.insert(3, "cf_idx", cf_idx)
                group = group.drop(["range_lost", "target_change", "cf_group"],axis=1)
                if self.exclude_cf is not None and cf_idx in self.exclude_cf:
                    continue
                if self.test_cf is not None and cf_idx in self.test_cf:
                    test_events.append(self.compute_cf_features(group))
                else:
                    cf_events.append(self.compute_cf_features(group))
        
        self.gmm_dfs = [[] for _ in range(self.n_cluster)]
        for i in range(self.n_cluster):
            for df in cf_events:
                idx_arr = (df["rank_0_cluster"]==i) | (df["rank_1_cluster"]==i)
                d = df[idx_arr]
                if len(d) > 0:
                    self.gmm_dfs[i].append(d)
        
        print(f"Processed {len(cf_events)} cf_events")
        concat_df = pd.concat(cf_events)
        concat_df.to_csv(self.CF_CSV)
        concat_df.to_pickle(self.CF_PKL)
        print("Saved Car Following File")
        return cf_events

    # -------------------------------
    #
    # LOAD
    # 
    # -------------------------------
    def load_cong(self):
        """load processed congestion dataset"""
        if not os.path.isfile(self.CONG_CSV) \
            and not os.path.isfile(self.CONG_PKL):
            raise FileExistsError("No processed Congestion dataset found, process before use!")
        jams = []
        try:
            cong_df = pd.read_pickle(self.CONG_PKL, index_col=0)
        except:
            cong_df = pd.read_csv(self.CONG_CSV, index_col=0)
        for _, group in cong_df.groupby('congestion'):
            jams.append(group)
        return jams

    def load_cf(self):
        """load processed car following events dataset"""
        if not os.path.isfile(self.CF_CSV) \
            and not os.path.isfile(self.CF_PKL):
            raise FileExistsError("No processed CF dataset found, process before use!")
        jams = []
        try:
            cf_df = pd.read_pickle(self.CF_PKL, index_col=0)
        except:
            cf_df = pd.read_csv(self.CF_CSV, index_col=0)
        for _, group in cf_df.groupby("cf_idx"):
            jams.append(group)
        return jams


    # -------------------------------
    #
    # ITERATORS
    # 
    # -------------------------------
    @property
    def __cfs(self):
        """iterator for jams"""
        for df in self.cf_events:
            if df.loc[df.index[0],"type"] == self.TYPE_JAM:
                yield df
            else:
                continue
    # @property
    # def __pre_cfs(self):
    #     """iterator for pre jams"""
    #     for df in self.cf_events:
    #         if df.loc[df.index[0],"type"] == self.TYPE_PRE:
    #             yield df
    # @property
    # def __post_cfs(self):
    #     """iterator for post jams"""
    #     for df in self.cf_events:
    #         if df.loc[df.index[0],"type"] == self.TYPE_POST:
    #             yield df
    @property
    def __congests(self):
        """iterator for jams"""
        for df in self.congs:
            if df.loc[df.index[0],"type"] == self.TYPE_JAM:
                yield df
    # @property
    # def __pre_congests(self):
    #     """iterator for pre jams"""
    #     for df in self.congs:
    #         if df.loc[df.index[0],"type"] == self.TYPE_PRE:
    #             yield df
    # @property
    # def __post_congests(self):
    #     """iterator for post jams"""
    #     for df in self.congs:
    #         if df.loc[df.index[0],"type"] == self.TYPE_POST:
    #             yield df

    # -------------------------------
    #
    # COMPUTING ADDITIONAL FEATURES
    # 
    # -------------------------------
    @classmethod
    def compute_cong_features(cls, df):
        # Make time in unit of seconds
        df.loc[:,"time"] *= 0.01    
        next_col = AutoIncrementer(df.shape[1])
        
        # Calibrate IMU Acceleration
        df.insert(next_col(), "calibratedAx", cls.calibrate_Ax(df))

        # Add time since congestion
        df.insert(next_col(), "timeSinceCongestion", df.loc[:, "time"]-df.loc[df.index[0],"time"])

        # Add Time Headway
        df.insert(next_col(), "timeHeadway", df.loc[:, "range"].div(df.loc[:, "speed"]).replace(np.inf, 99))
        return df
    
    @classmethod
    def compute_cf_features(cls, df):
        next_col = AutoIncrementer(df.shape[1])

        # Add Visual Looming Information
        W = 2    # Car with as 2 meters by default
        df.insert(next_col(), "VL_theta", cls.calc_vl_theta(df, W))
        df.insert(next_col(), "VL_dtheta", cls.calc_vl_dtheta(df, W))

        # Add computed lead vehicle information
        df.insert(next_col(), "lv_speed", cls.calc_lv_speed(df))
        df.insert(next_col(), "lv_Ax", cls.calc_lv_acc(df))

        # Calibrate distance
        df.insert(next_col(), "distanceCalibrated", cls.normalize_dist(df))

        # Add lead vehicle distance
        df.insert(next_col(), "lv_distance", cls.calc_lv_x(df)), 

        # Adjust data for CF
        df.insert(next_col(), "timeSinceCF", cls.calc_time_since_cf(df))
        df.insert(next_col(), "distanceSinceCF", cls.calc_x_since_cf(df))
        df.insert(next_col(), "lv_distanceSinceCF", cls.calc_lv_x_since_cf(df))
        return df

    @staticmethod
    def calibrate_Ax(df):
        imu_mean_shift = df.loc[df["speed"]==0, "Ax"].mean()
        if np.isnan(imu_mean_shift).any():
            imu_mean_shift = 0.0
        return (df["Ax"]-imu_mean_shift)

    @staticmethod
    def calc_vl_theta(df, W) -> np.ndarray:
        """computes visual looming theta"""
        return 2 * np.arctan2(W, 2*df.loc[:, "range"])

    @staticmethod               
    def calc_vl_dtheta(df, W) -> np.ndarray:
        """compute visual looming delta theta"""
        return -W * df.loc[:,"rangerate"] / (df.loc[:,'range']**2 + W**2 / 4)
    
    @staticmethod
    def calc_lv_speed(df) -> pd.Series:
        """Computes lead vehicle speed from ego speed and rangerate"""
        return (df.loc[:, "speed"] + df.loc[:, "rangerate"])

    @classmethod
    def calc_lv_acc(cls, df) -> pd.Series:
        """Computes lead vehicle acceleration using a moving average filter."""
        # GPT generated MAF
        lv_speed = df["lv_speed"]
        lv_speed_diff = lv_speed.diff()
        # Compute rolling mean of speed differences over a window of 5 timestamps
        rolling_mean_diff = lv_speed_diff.rolling(window=10, min_periods=1).mean()
        return np.nan_to_num(rolling_mean_diff/cls.TICK, 0)
    
    @classmethod
    def normalize_dist(cls, df) -> np.ndarray:
        # Get total distance travelled over time
        raw_dist = df.loc[:, "distance"].to_numpy()
        delta_dist = raw_dist[-1] - raw_dist[0]

        # If didn't even move, just not calibrate
        if delta_dist < 1:
            return df.loc[:, "distance"]

        # Compute distance by speed
        # since acceleration is not reliable, we don't consider it
        tick_dist = df.loc[:,"speed"].to_numpy() * cls.TICK
        tick_dist[0] += raw_dist[0]
        comp_dist = tick_dist.cumsum()
        delta_comp_dist = comp_dist[-1] - comp_dist[0]

        # Compute ratio, and modify comp_dist to normalize
        ratio = delta_dist / delta_comp_dist
        comp_dist *= ratio
        return comp_dist
    
    @staticmethod
    def calc_lv_x(df):
        """calculates distance travelled by lead vehicle since start of ticking"""
        return (df.loc[:, "distanceCalibrated"] + df.loc[:, "range"])
    
    @staticmethod
    def calc_time_since_cf(df):
        """calculates distance travelled since car following"""
        baseline = df.iloc[0]["timeSinceCongestion"]
        return df["timeSinceCongestion"]-baseline
    
    @staticmethod
    def calc_x_since_cf(df) -> pd.Series:
        """calculates distance travelled since car following"""
        baseline = df.iloc[0]["distanceCalibrated"]
        return df["distanceCalibrated"] - baseline
    
    @staticmethod
    def calc_lv_x_since_cf(df):
        """calculates distance travelled since car following"""
        baseline = df.iloc[0]["distanceCalibrated"]
        return df["lv_distance"] - baseline
    
    # -------------------------------
    #
    # PLOTTING DATA
    # 
    # -------------------------------
    @staticmethod
    def lim2digits(x):
        return "{:.2f}".format(x)
    
#     def summarize(self):
#         # Number of congestions
#         print("Dataset Summary:\n")
        
#         # Congestions
#         print(f"- A total of {sum(1 for s in self.congs)} congestions")
#         congest_lens = []
#         for i, (prefix, l) in enumerate(zip(["", "pre-", "post-"], [self.congests, self.pre_congests, self.post_congests])):
#             congest_lens.append(sum(len(df) for df in l))
#             print(f"- {len(l)} {prefix}congestions with {congest_lens[i]} data points")
#         print("")
        
#         # Congestions
#         print(f"- A total of {sum(1 for s in self.cf_events)} car following evengts")
#         cf_lens = []
#         for i, (prefix, l) in enumerate(zip(["", "pre-", "post-"], [self.cfs, self.pre_cfs, self.post_cfs])):
#             cf_lens.append(sum(len(df) for df in l))
#             print(f"- {len(l)} {prefix}cf events with {cf_lens[i]} data points")
#         print("")
        
#         # Congestions
#         print(f"- A total of {self.lim2digits(sum(cf_lens)/sum(congest_lens)*100)}% are car following")
#         for i, prefix in enumerate(["", "pre-", "post-"]):
#             print(f"- {self.lim2digits(cf_lens[i]/congest_lens[i]*100)} are car following {prefix}jams")
#         print("")
        
                
    def plot(self, out_dir, sub_dirs=['jam','pre', 'post']):
        for sub_dir in sub_dirs:
            sd = os.path.join(out_dir, sub_dir)
            if not os.path.isdir(sd):
                os.makedirs(sd)
                
        if "jam" in sub_dirs:
            for df in self.cfs:
                self.plot_single_cf(df, os.path.join(out_dir, "jam", str(df.loc[df.index[0],"cf_idx"].item())+"_summary.png"))
        # if "pre" in sub_dirs:
        #     for df in self.pre_cfs:
        #         self.plot_single_cf(df, os.path.join(out_dir, "pre", str(df.loc[df.index[0],"cf_idx"].item())+"_summary.png"))
        # if "post" in sub_dirs:
        #     for df in self.post_cfs:
        #         self.plot_single_cf(df, os.path.join(out_dir, "post", str(df.loc[df.index[0],"cf_idx"].item())+"_summary.png"))
        print(f"Dataset items plotted at {out_dir}")
        
        
    def plot_single_cf(self, df, out_file):
        """generate a plot for a single cf event"""
        num_subplots = 5
        fig, axs = plt.subplots(num_subplots, 1, figsize=(8, 4*num_subplots))
        
        # Plot individual df
        self.plot_t_acc(axs[0], df)  # Acceleration
        self.plot_t_v(axs[1], df)    # speed
        self.plot_t_x(axs[2], df)    # Location
        self.plot_t_s(axs[3], df)    # Space headway
        self.plot_v_s(axs[4], df)    # speed against headway
        
        plt.suptitle(f"Summary of Driver {df.loc[df.index[0],'driver']} Trip {df.loc[df.index[0],'trip']} Cong {df.loc[df.index[0],'congestion']}")
        plt.savefig(out_file)
        plt.close()
        

    def plot_t_acc(self, ax, df):
        """Plot ego and lv acceleration"""
        ax.plot(df.loc[:, "timeSinceCF"], df.loc[:, "Ax"], label="ego acceleration")
        ax.plot(df.loc[:, "timeSinceCF"], df.loc[:, "lv_Ax"], label="lv acceleration")
        ax.legend()

    def plot_t_v(self, ax, df):
        """Plot ego and lv acceleration"""
        ax.plot(df.loc[:, "timeSinceCF"], df.loc[:, "speed"], label="ego speed")
        ax.plot(df.loc[:, "timeSinceCF"], df.loc[:, "lv_speed"], label="lv speed")
        ax.legend()

    def plot_t_x(self, ax, df):
        """Plot ego distance"""
        ax.plot(df.loc[:, "timeSinceCF"], df.loc[:, "distanceSinceCF"], label="ego dist since CF")
        ax.plot(df.loc[:, "timeSinceCF"], df.loc[:, "lv_distanceSinceCF"], label="lv dist since CF")
        ax.legend()
        
    def plot_t_s(self, ax, df):
        """Plot ego and lv space headway"""
        ax.plot(df.loc[:, "timeSinceCF"], df.loc[:, "range"], label="space headway")
        ax.legend()
    
    def plot_v_s(self, ax, df):
        """Plot ego and lv space headway"""
        ax.scatter(df.loc[:, "speed"], df.loc[:, "range"], label="ego speed vs range")
        ax.legend()




cf_failure_dict = {
    0: [[1700, 1717.5]],
    4: [[2260, 2275]],
    7: [[1550, 1585]],
    8: [[1610, 1750]],
    9: [[2715, 2740], [2750, 2770]],
    10: [[2790, 2860]], 
    11: [[2870, 2890]],
    14: [[1430, 1460]],
    15: [[2440, 2750]], 
    16: [[2760, 2880]],
    17: [[5040, 5060]],
    21: [[160, 180], [280, 300]],
    22: [[1445, 1460], [1465, 1540]],
    # Distracted -- 23: [[]],
    24: [[1150, 1200], [1230, 1280], [1300, 1330]], 
    # Good -- 25
    # Distraction -- 26, 27, 28, 29
    30: [[1200, 1250]],
    31: [[2900, 3020]],
    # Distraction -- 32
    33: [[3100, 3250]],
    34: [[2520, 2600]],
    35: [[4940, 4960], [4970, 4980]],
    36: [[4985, 5010]],
    40: [[5170, 5200], [5330, 5370], [5400, 5800]],
    42: [[5960, 6150]],
    43: [[6560, 6590]],
    45: [[6670, 6685]],
    46: [[2080, 2140]],
    47: [[2090, 2140], [2140, 2180]],
    48: [[630, 660], [730, 780]],
    49: [[820, 900]],
    50: [[3140, 3155], [3165, 3190], [3230, 3260]],
    51: [[3280, 3290]],
    52: [[3305, 3320]],
    53: [[2830, 2880], [2910, 2940]],
    54: [[2930, 2950]], 
    55: [[1700,1750]],
    57: [[1790, 1850], [1850, 1920]],
    58: [[1920, 1960]],
    # Done
}


class ParamDL2Chunks:
    # Example dictionary of shorthands and full descriptions
    shorthand_dict = {
        't': 'time',
        's': 'space gap',
        'v': 'speed',
        'relv': 'relative velocity',
        'a': 'acceleration',
        'ap': 'acceleration pedal',
        'bp': 'brake pedal',
    }

    def __init__(self, dataset, ranges, chunk_len, interval, data_fields=["calibratedAx", "range", "speed", "rangerate", "accelpedal", "brakepedal"]):
        self.dataset = dataset
        self.data_fields = data_fields
        self.chunk_len = chunk_len
        self.interval = interval
        self.cluster_colors = ['r','y','c','g','m','b','k']
        for k in ranges.keys():
            assert(k >= 0 and k <= len(dataset.cf_events))

        # Get Ranges
        self.ranges = ranges
        self.ranged_dfs = self.read_limits(dataset.cf_events, ranges)
        self.check_num(self.ranged_dfs, ranges)

        # Sample
        self.sampled_chunks, self.sampled_times, self.sampled_meta = self.sample(self.ranged_dfs, chunk_len, interval)
        self.sampled_chunks_stacked = np.array(self.sampled_chunks)
        self.sampled_times_stacked = np.array(self.sampled_times)
        assert(self.sampled_chunks_stacked.shape == (len(self.sampled_chunks), chunk_len, len(data_fields)))
        # Finish

    def read_limits(self, dfs, ranges):
        ranged_dfs = []
        for k, rg_list in ranges.items():
            df = dfs[k]
            for rg in rg_list:
                ranged_dfs.append(df[(df['time'] >= rg[0]) & (df['time'] <= rg[1])])
        return ranged_dfs
    
    def check_num(self, dfs, ranges):
        num_ranges = sum([len(l) for l in ranges.values()])
        num_dfs = len(dfs)
        assert(num_ranges == num_dfs)

    def sample(self, dfs, chunk_len, interval):
        samples = []
        sample_times = []
        sample_meta = []
        for df in dfs:
            if len(df) < chunk_len:
                print(f"Encountered Sequence with length {len(df)} < Required {chunk_len}")
                pass
            for i in range(0, len(df)-chunk_len, interval):
                samp = df.loc[df.index[i]:df.index[i+chunk_len-1], self.data_fields].to_numpy()
                samp_time = df.loc[df.index[i]:df.index[i+chunk_len-1], "time"].to_numpy()
                assert(samp.shape[0] == chunk_len)
                samples.append(samp)
                sample_times.append(samp_time)
                sample_meta.append(df.loc[df.index[0], "cf_idx"])
        print(f"Extracted {len(samples)} samples")
        return samples, sample_times, sample_meta

    def kmeans_dtw_cluster(self, n_clusters):
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw",
                                max_iter=50, random_state=42, dtw_inertia=True)
        model.fit(self.sampled_chunks_stacked)
        return model

    def calculate_wcss(self, max_clusters):
        wcss = []
        for n_clusters in range(2, max_clusters + 1):
            model = self.kmeans_dtw_cluster(n_clusters)
            wcss.append(model.inertia_)
        return wcss
    
    def plot_elbow_method(self, wcss):
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, len(wcss) + 2), wcss, marker='o', linestyle='--')
        plt.title('Elbow Method For Optimal Number of Clusters')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    def fit(self, n_clusters):
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw",
                                max_iter=10, random_state=42, dtw_inertia=True)
        labels = model.fit_predict(self.sampled_chunks_stacked)
        # Show results
        print(f"There are {[(labels==i).sum() for i in range(n_clusters)]}")

        subgroups = [self.sampled_chunks_stacked[labels==i, ...] for i in range(n_clusters)]
        # nField x 4 x (nSeq x nTime)
        field_clusters = [[subgroups[j][...,i] for j in range(n_clusters)] for i in range(subgroups[0].shape[-1])]
        return model, labels, field_clusters
    

    def plot_pedal_percentage(self, field_clusters):
        n_clusters = len(field_clusters[0])
        """field_clusters: nField x nCluster x T"""
        acc_cluster = field_clusters[-2]
        dec_cluster = field_clusters[-1]
        acc_ratio = np.array([(c > 0).sum() / c.size for c in acc_cluster])
        brake_ratio = np.array([(c > 0).sum() / c.size for c in dec_cluster])
        non_ratio = np.ones_like(brake_ratio) - acc_ratio - brake_ratio

        fig, ax = plt.subplots(figsize=(16,5))
        # Bar heights
        bars = [f'Cluster {i}' for i in range(1, n_clusters+1)]
        bar_height = 0.5
        # Positions for the bars
        bar_positions = np.arange(len(bars))

        # Plot bars
        acc_bars = ax.barh(bar_positions, acc_ratio, height=bar_height, label='Accelerator Applied', alpha=0.8)
        brake_bars = ax.barh(bar_positions, brake_ratio, height=bar_height, left=acc_ratio, label='Brake Applied', alpha=0.8)
        non_bars = ax.barh(bar_positions, non_ratio, height=bar_height, left=acc_ratio + brake_ratio, label='Neither', alpha=0.8)

        # Add text annotations for percentage values
        for bar in acc_bars:
            width = bar.get_width()
            ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2, f'{width:.2%}', ha='center', va='center', color='white', fontsize=14)

        for bar in brake_bars:
            width = bar.get_width()
            left = bar.get_x()
            ax.text(left + width / 2, bar.get_y() + bar.get_height() / 2, f'{width:.2%}', ha='center', va='center', color='white', fontsize=14)

        for bar in non_bars:
            width = bar.get_width()
            left = bar.get_x()
            ax.text(left + width / 2, bar.get_y() + bar.get_height() / 2, f'{width:.2%}', ha='center', va='center', color='white', fontsize=14)
            
        # Set labels and title
        ax.set_yticks(bar_positions)
        ax.set_yticklabels(bars)
        ax.set_xlabel('Proportion')
        ax.set_title('Proportion of Pedal Conditions')
        for tick_label, color in zip(ax.get_yticklabels(), self.cluster_colors[:n_clusters]):
            dim_color = mcolors.to_rgba(color, alpha=0.8)
            tick_label.set_color(dim_color)

        plt.legend()
        plt.savefig("results/cluster4_t10/c4t10_pedal_percentage.png")
        plt.show()
        plt.close()


    def plot_time_with_centroid(self, model, field_clusters, field_labels):
        """field_clusters: list of nField lists, each containing nCluster NP arrays of shape (nSequence, Time)"""
        nCluster = len(field_clusters[0])

        for field_idx, clusters in enumerate(field_clusters):
            fig, axes = plt.subplots(1, nCluster, figsize=(6 * nCluster, 6))

            if nCluster == 1:
                axes = [axes]

            for cluster_idx, cluster_data in enumerate(clusters):
                for sequence in cluster_data:
                    axes[cluster_idx].plot(sequence, alpha=0.3)
                axes[cluster_idx].plot(model.cluster_centers_[cluster_idx, :, field_idx], color='r', linewidth=4)
                axes[cluster_idx].set_title(f'Cluster {cluster_idx + 1}', fontsize=30)
                axes[cluster_idx].set_xlabel('Time', fontsize=30)
                axes[cluster_idx].set_ylabel('Value', fontsize=30)
                facecolor = mcolors.to_rgba(self.cluster_colors[cluster_idx], alpha=0.1)
                axes[cluster_idx].set_facecolor(facecolor)

            plt.suptitle(f"{field_labels[field_idx]} vs Time Plot", fontsize=32)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"results/cluster4_t10/field_{field_idx + 1}_clusters.png")
            plt.show()
            plt.close()

    def plot_dist(self, field_clusters, labels):
        for field_idx, clusters in enumerate(field_clusters):
            fig, ax = plt.subplots()
            fig.suptitle(f'{labels[field_idx]} Distribution')

            ax.boxplot([arr.flatten() for arr in clusters], vert=False)
            ax.set_xlabel(labels[field_idx])
            plt.tight_layout()
            plt.show()
            plt.close()

    def plot_scatter_pairs(self, field_clusters, labels):
        """
        Plots scatterplots for all unique pairs of given arrays.

        Parameters:
        arrays (list of 2D arrays): (nSeq)x(T)
        labels: (nArrays) length list
        """
        # Plot val against one another
        n = len(labels)

        for i in range(n):
            for j in range(i+1, n):
                fig, ax = plt.subplots()
                x_seqs = [f.flatten() for f in field_clusters[i]]
                y_seqs = [f.flatten() for f in field_clusters[j]]
                for m in range(len(x_seqs)):
                    ax.scatter(x_seqs[m], y_seqs[m], s=0.5, color=self.cluster_colors[m])
                ax.set_xlabel(labels[i])
                ax.set_ylabel(labels[j])
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_regions(self, model, n_cluster):
        duration = self.chunk_len / 10
        for i, df in enumerate(self.dataset.cf_events):
            # fig, ax = plt.subplots(figsize=(8,4))
            if i not in self.ranges.keys():
                continue
            c
            # sample
            cluster_times = [[] for _ in range(n_cluster)]
            for rg in self.ranges[i]:
                s = rg[0]
                e = rg[1]
                s_t = s
                print(duration)
                while s_t + duration < e:
                    sample = df.loc[(df["time"] >= s_t) & (df["time"] < s_t+duration), self.data_fields].to_numpy().reshape(1,-1, len(self.data_fields))
                    if sample.shape[1] != self.chunk_len:
                        break
                    label = model.predict(sample)
                    cluster_times[label[0]].append((s_t, s_t + duration))
                    s_t += duration

            plt.plot(df["time"], df["range"])
            for c_idx, times in enumerate(cluster_times):
                for time in times:
                    print(time)
                    plt.axvspan(time[0], time[1], color=self.cluster_colors[c_idx], alpha=0.2, lw=0)
            plt.tight_layout()
            plt.savefig(f"results/cluster6_f2_t10_colored_s/cluster_colored_{i+1}.png")
            plt.close()