import pandas as pd
import numpy as np
import glob
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from datetime import datetime as dt
import math
import seaborn as sns
import ruptures

MAC_TO_EPOCH_OFFSET = 2082844800


def reprocess_binary_data_sig(bin_file_path, log_start_datetime, sensitivity=1, bckgrnd=0, data_freq=10, skip_start=0
                          , skip_end=0, n_bad_online=0, dtype='>i2'):
    epoch_time = (dt.strptime(log_start_datetime, '%d/%m/%Y %H:%M:%S') - dt.strptime('01/01/1904',
                                                                                     '%d/%m/%Y')).total_seconds()
    # epoch time is the number of seconds between 01/01/1904 and the log_start_time

    file_list = [f for f in listdir(bin_file_path) if isfile(join(bin_file_path, f))]
    # Produces a list of filenames based off the directory specified

    filt_file_list = file_list[skip_start: len(file_list) - skip_end]  # function to skip files

    channel_format = {'sig_counts': 0, 'ref_counts': 1, 'seed_LD_current': 2, 'laser_pwr_PT0': [3, 4], 'time_ms': [7, 8]
                      , 'seed_LD_mode': 9}
    channel_count = 10
    # channel_format is a dict of the indices for the different data columns (those with two indices have been
    # split into hi lo numbers

    for file in filt_file_list:

        print('\nReprocessing file %s' % file)
        # to let you know which file is being looked at

        data = np.fromfile(bin_file_path + '/' + file, dtype=dtype)

        frames = np.array(data)
        decimate_arr = [frames[idx::channel_count] for idx in range(channel_count)]

        binary_data_dict = {}

        for dict_key in channel_format.items():
            binary_data_dict[dict_key[0]] = []

        for channel, channel_ID in channel_format.items():
            if type(0) == type(channel_ID):
                binary_data_dict[channel] = np.concatenate((binary_data_dict[channel], decimate_arr[channel_ID]))
            if type([]) == type(channel_ID):
                hi = decimate_arr[channel_ID[0]] * 65536
                lo = np.where(decimate_arr[channel_ID[1]] < 0
                              , decimate_arr[channel_ID[1]] + 65536, decimate_arr[channel_ID[1]])
                binary_data_dict[channel] = np.concatenate((binary_data_dict[channel], lo + hi))
        
        # linearise counts
        rep_rate = 200000
        sample_rate = 100
        max_counts = rep_rate / sample_rate

        binary_data_dict['sig_counts_lin'] = np.log(1 - (binary_data_dict['sig_counts']/max_counts)) * -max_counts
        
        
        # normalise counts
        x = binary_data_dict['laser_pwr_PT0'] / 100000

        binary_data_dict['sig_counts_norm'] = binary_data_dict['sig_counts_lin'] / (-0.06468598076441119*np.exp(-0.45385800649400554*x) + 0.06440536807374303)
        #((-0.0037064580782284373*x**2) + (0.02673116696251598*x) + 0.0001007672398719008)

        reprocessed_data_dict = {}

        keys = ['Time_ms', 'SO2_mr', 'On_cts_time_ms', 'On_cts', 'On_cts_norm', 'On_lsr_power_V'
                , 'Off_cts_time_ms', 'Off_cts', 'Off_cts_norm', 'Off_lsr_power_V', 'Cts_diff']

        for key in keys:
            reprocessed_data_dict[key] = []

        group_avg = int((1 / data_freq) * 10)
        skip_set = (100 * group_avg) - 100  # skip_set = 100 yields 5 Hz data

        dir_path = os.path.dirname(__file__)

        path = r'{}/processed data'.format(dir_path)

        try:
            os.makedirs(path)
        except OSError:
            pass

        binary_file = open('processed data/%s_SO2_processed_data_%.0f.txt'
                           % (file.split(sep="_")[1].split(sep=" ")[0], file_list.index(file)), 'w+')
        binary_file.write('time_ms,Cts_diff,off_cts_norm,on_cts_norm,off_lsr_pwr_V,on_lsr_pwr_V\n')
        #binary_file.write('time_off,off_counts,off_counts_norm,laser_pwr_off\n')

        tot_steps = len(binary_data_dict['time_ms']) - ((10 * group_avg) + 1)

        SO2_mr = []
        cts_diff = []
        off_counts_norm = []
        off_counts = []
        laser_pwr_off = []
        on_counts_norm = []
        on_counts = []
        laser_pwr_on = []
        bin_time = [0]
        time_on = []
        time_off = []
        mac_time = []

        for i in range(len(binary_data_dict['time_ms']) - ((10 * group_avg) + 1)):
            print('\r%.2f' % (abs(1 - (tot_steps - i) / tot_steps) * 100), end='')

            curr_time = binary_data_dict['time_ms'][i] + (epoch_time * 1000)

            if binary_data_dict['seed_LD_mode'][i] == 5:
                time_off.append(curr_time)
                off_counts_norm.append(binary_data_dict['sig_counts_norm'][i])
                off_counts.append(binary_data_dict['sig_counts_lin'][i])
                laser_pwr_off.append(binary_data_dict['laser_pwr_PT0'][i])

            if binary_data_dict['seed_LD_mode'][i] == 6:
                time_on.append(curr_time)
                on_counts_norm.append(binary_data_dict['sig_counts_norm'][i])
                on_counts.append(binary_data_dict['sig_counts_lin'][i])
                laser_pwr_on.append(binary_data_dict['laser_pwr_PT0'][i])


            if binary_data_dict['seed_LD_mode'][i] == 6 and binary_data_dict['seed_LD_mode'][i + 1] == 5:
                if binary_data_dict['time_ms'][i] - bin_time[-1] != skip_set:

                    bin_time.append(binary_data_dict['time_ms'][i])
                    mac_time.append(curr_time)

                    on_cts = []
                    on_lsr_pwr = []
                    for j in range(group_avg):
                        on_cts.append(binary_data_dict['sig_counts_norm'][i - (7 - n_bad_online) - (j * 10): i + 1 - (j * 10)])
                        on_lsr_pwr.append(binary_data_dict['laser_pwr_PT0'][i - (7 - n_bad_online) - (j * 10): i + 1 - (j * 10)])

                    off_cts = []
                    off_lsr_pwr = []
                    for j in range(group_avg):
                        off_cts.append(binary_data_dict['sig_counts_norm'][i + 1 - (j * 10): i + 3 - (j * 10)])
                        off_lsr_pwr.append(binary_data_dict['laser_pwr_PT0'][i + 1 - (j * 10): i + 3 - (j * 10)])

                    #set_len = (group_avg * len(on_cts[0])) + (group_avg * len(off_cts[0]))
                    set_len = 10

                    mix_r = ((((np.mean(on_cts) - np.mean(off_cts)) * set_len) - (bckgrnd / data_freq))
                             / (sensitivity * (((1 / data_freq) * 1000) / 1000)))
                    cts = (np.mean(on_cts) - np.mean(off_cts)) * set_len * data_freq

                    SO2_mr.append(mix_r)
                    cts_diff.append(cts)

        #for j in range(len(off_counts)):
            #binary_file.write(str(time_off[j]) + ',')
            #binary_file.write(str(off_counts[j]) + ',')
            #binary_file.write(str(off_counts_norm[j]) + ',')
            #binary_file.write(str(laser_pwr_off[j]) + '\n')

                    if not math.isnan(mix_r):

                        binary_file.write(str(curr_time) + ',')
                        binary_file.write(str(cts) + ',')
                        binary_file.write(str(np.mean(off_cts)) + ',')
                        binary_file.write(str(np.mean(on_cts)) + ',')
                        binary_file.write(str(np.mean(off_lsr_pwr)) + ',')
                        binary_file.write(str(np.mean(on_lsr_pwr)) + '\n')


def reprocess_binary_data_ref(bin_file_path, log_start_datetime, sensitivity=1, bckgrnd=0, data_freq=10, skip_start=0
                          , skip_end=0, n_bad_online=0, dtype='>i2'):
    epoch_time = (dt.strptime(log_start_datetime, '%d/%m/%Y %H:%M:%S') - dt.strptime('01/01/1904',
                                                                                     '%d/%m/%Y')).total_seconds()
    # epoch time is the number of seconds between 01/01/1904 and the log_start_time

    file_list = [f for f in listdir(bin_file_path) if isfile(join(bin_file_path, f))]
    # Produces a list of filenames based off the directory specified

    filt_file_list = file_list[skip_start: len(file_list) - skip_end]  # function to skip files

    channel_format = {'sig_counts': 0, 'ref_counts': 1, 'seed_LD_current': 2, 'laser_pwr_PT0': [3, 4], 'time_ms': [7, 8]
                      , 'seed_LD_mode': 9}
    channel_count = 10
    # channel_format is a dict of the indices for the different data columns (those with two indices have been
    # split into hi lo numbers

    for file in filt_file_list:

        print('\nReprocessing file %s' % file)
        # to let you know which file is being looked at

        data = np.fromfile(bin_file_path + '/' + file, dtype=dtype)

        frames = np.array(data)
        decimate_arr = [frames[idx::channel_count] for idx in range(channel_count)]

        binary_data_dict = {}

        for dict_key in channel_format.items():
            binary_data_dict[dict_key[0]] = []

        for channel, channel_ID in channel_format.items():
            if type(0) == type(channel_ID):
                binary_data_dict[channel] = np.concatenate((binary_data_dict[channel], decimate_arr[channel_ID]))
            if type([]) == type(channel_ID):
                hi = decimate_arr[channel_ID[0]] * 65536
                lo = np.where(decimate_arr[channel_ID[1]] < 0
                              , decimate_arr[channel_ID[1]] + 65536, decimate_arr[channel_ID[1]])
                binary_data_dict[channel] = np.concatenate((binary_data_dict[channel], lo + hi))
        
        # linearise counts
        rep_rate = 200000
        sample_rate = 100
        max_counts = rep_rate / sample_rate

        binary_data_dict['ref_counts_lin'] = np.log(1 - (binary_data_dict['ref_counts']/max_counts)) * -max_counts
        
        
        # normalise counts
        x = binary_data_dict['laser_pwr_PT0'] / 100000

        binary_data_dict['ref_counts_norm'] = binary_data_dict['ref_counts_lin'] / (-0.06468598076441119*np.exp(-0.45385800649400554*x) + 0.06440536807374303)

        reprocessed_data_dict = {}

        keys = ['Time_ms', 'SO2_mr', 'On_cts_time_ms', 'On_cts', 'On_cts_norm', 'On_lsr_power_V'
                , 'Off_cts_time_ms', 'Off_cts', 'Off_cts_norm', 'Off_lsr_power_V', 'Cts_diff']

        for key in keys:
            reprocessed_data_dict[key] = []

        group_avg = int((1 / data_freq) * 10)
        skip_set = (100 * group_avg) - 100  # skip_set = 100 yields 5 Hz data

        dir_path = os.path.dirname(__file__)

        path = r'{}/processed data'.format(dir_path)

        try:
            os.makedirs(path)
        except OSError:
            pass

        binary_file = open('processed data/%s_SO2_processed_data_%.0f.txt'
                           % (file.split(sep="_")[1].split(sep=" ")[0], file_list.index(file)), 'w+')
        binary_file.write('time_ms,Cts_diff,off_cts_norm,on_cts_norm,off_lsr_pwr_V,on_lsr_pwr_V\n')
        #binary_file.write('time_off,off_counts,off_counts_norm,laser_pwr_off\n')

        tot_steps = len(binary_data_dict['time_ms']) - ((10 * group_avg) + 1)

        SO2_mr = []
        cts_diff = []
        off_counts_norm = []
        off_counts = []
        laser_pwr_off = []
        on_counts_norm = []
        on_counts = []
        laser_pwr_on = []
        bin_time = [0]
        time_on = []
        time_off = []
        mac_time = []

        for i in range(len(binary_data_dict['time_ms']) - ((10 * group_avg) + 1)):
            print('\r%.2f' % (abs(1 - (tot_steps - i) / tot_steps) * 100), end='')

            curr_time = binary_data_dict['time_ms'][i] + (epoch_time * 1000)

            if binary_data_dict['seed_LD_mode'][i] == 5:
                time_off.append(curr_time)
                off_counts_norm.append(binary_data_dict['ref_counts_norm'][i])
                off_counts.append(binary_data_dict['ref_counts_lin'][i])
                laser_pwr_off.append(binary_data_dict['laser_pwr_PT0'][i])

            if binary_data_dict['seed_LD_mode'][i] == 6:
                time_on.append(curr_time)
                on_counts_norm.append(binary_data_dict['ref_counts_norm'][i])
                on_counts.append(binary_data_dict['ref_counts_lin'][i])
                laser_pwr_on.append(binary_data_dict['laser_pwr_PT0'][i])


            if binary_data_dict['seed_LD_mode'][i] == 6 and binary_data_dict['seed_LD_mode'][i + 1] == 5:
                if binary_data_dict['time_ms'][i] - bin_time[-1] != skip_set:

                    bin_time.append(binary_data_dict['time_ms'][i])
                    mac_time.append(curr_time)

                    on_cts = []
                    on_lsr_pwr = []
                    for j in range(group_avg):
                        on_cts.append(binary_data_dict['ref_counts_norm'][i - (7 - n_bad_online) - (j * 10): i + 1 - (j * 10)])
                        on_lsr_pwr.append(binary_data_dict['laser_pwr_PT0'][i - (7 - n_bad_online) - (j * 10): i + 1 - (j * 10)])

                    off_cts = []
                    off_lsr_pwr = []
                    for j in range(group_avg):
                        off_cts.append(binary_data_dict['ref_counts_norm'][i + 1 - (j * 10): i + 3 - (j * 10)])
                        off_lsr_pwr.append(binary_data_dict['laser_pwr_PT0'][i + 1 - (j * 10): i + 3 - (j * 10)])

                    #set_len = (group_avg * len(on_cts[0])) + (group_avg * len(off_cts[0]))
                    set_len = 10

                    mix_r = ((((np.mean(on_cts) - np.mean(off_cts)) * set_len) - (bckgrnd / data_freq))
                             / (sensitivity * (((1 / data_freq) * 1000) / 1000)))
                    cts = (np.mean(on_cts) - np.mean(off_cts)) * set_len * data_freq

                    SO2_mr.append(mix_r)
                    cts_diff.append(cts)

        #for j in range(len(off_counts)):
            #binary_file.write(str(time_off[j]) + ',')
            #binary_file.write(str(off_counts[j]) + ',')
            #binary_file.write(str(off_counts_norm[j]) + ',')
            #binary_file.write(str(laser_pwr_off[j]) + '\n')

                    if not math.isnan(mix_r):

                        binary_file.write(str(curr_time) + ',')
                        binary_file.write(str(cts) + ',')
                        binary_file.write(str(np.mean(off_cts)) + ',')
                        binary_file.write(str(np.mean(on_cts)) + ',')
                        binary_file.write(str(np.mean(off_lsr_pwr)) + ',')
                        binary_file.write(str(np.mean(on_lsr_pwr)) + '\n')


def load_hk(path, so2_cylinder_conc):
    """
    Loads the housekeeping data and calculates the SO2 mixing ratio.

    Args:
        - path (str): The folder where the housekeeping files are kept.
        - so2_cylinder_conc (float): The SO2 cylinder concentration, used to 
          calculate the mixing ratio as: 
              mixing ratio = MFC_read / Cell_flow * so2_cylinder_conc

    Returns:

        A pandas DataFrame with columns: 
            - 'Task',
            - 'ZA_SB_MFC_Read'
            - 'Cal_SO2_MFC_Read'
            - 'Time_s'
            - 'Cell_flow'
            - 'SO2_mixing_ratio'
    """
    all_files = glob.glob(path + "/*.txt")
    dfs = [
        pd.read_csv(filename, sep="\s+", index_col=None, header=0)
        for filename in all_files
    ]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    
    df["Date_time"] = pd.to_datetime(
        df["Time_s"] - MAC_TO_EPOCH_OFFSET, unit="s"
    )
    df = df.set_index("Date_time")
    df = df.sort_values(by="Date_time")

    # Subset HK to columns interested in and add column with SO2 MR
    df = df[
        ["Task", "ZA_SB_MFC_Read", "Cal_SO2_MFC_Read", "Time_s", "Cell_Flow", "Signal_LIF_Counts", "SO2_ppt", "Inlet_ZA_MFC_Read", "SO2_Cal_MFC_Set_P", "Cal_SB_Vl", "Cell_Pressure", "Thermistor_5", "T_board_NLO", "Laser_Power_PT_0", "Signal_Dk_Counts", "Ref_Dk_Counts"]
    ]
    
    df["SO2_mr"] = (
        df["Cal_SO2_MFC_Read"] / df["Cell_Flow"] * so2_cylinder_conc
    )
    
    return df


def load_counts(path):
    all_files = glob.glob(path + "/*.txt")
    dfs = [
        pd.read_csv(filename, sep=",", index_col=None, header=0)
        for filename in all_files
    ]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    
    df["Date_time"] = pd.to_datetime(
        df["time_ms"] - (MAC_TO_EPOCH_OFFSET*1000), unit="ms"
    )
    df = df.set_index("Date_time")
    df = df.sort_values(by="Date_time")
    
    return df


def rle(x):
    """
    Calculates a run-length encoding for a given boolean vector x.

    In particular, if x is (True, True, False, False, True, True, True, False, True)
    then output  (1,    1,    0,     0,     2,    2,    2,    0,     3   )

    Args:
        x (np.array): Numpy array of booleans.

    Returns:
        A np.array of integers representing groupings.
    """
    # This line finds all timepoints where it changes from a False to a True,
    # i.e. it was a cal and now it isn't, or vice versa
    # The cumsum takes a counter so the first grouping is 1, then the
    # after the first change the values will be 2, 3 after the next change etc...
    y = (x != x.shift()).cumsum()
    # This just says that all times there isn't a cal (False = 0), set to group 0
    y[x == 0] = 0
    # The above 2 steps will have resulted in y values 0, 1, 3, 5, 7, etc...
    # since every other group (not cals) have been set to 0.
    # The following 2 lines just resets these labels into 0, 1, 2, 3, 4, 5 etc...
    y_cat = pd.Categorical(y, categories=sorted(y.unique()))
    y_relabelled = pd.Series(y_cat.codes).values
    return y_relabelled


def remove_small_groups(x, threshold):
    """
    Removes groups that do not meet a size criteria.

    A value of 0 indicates the default group.

    I.e. if x = [0, 1, 1, 1, 0, 2, 2, 0, 3, 3, 3, 3]
    and threshold = 3
    Then the output will be [0, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3]
    since the 2 group is removed and set to 0

    Args:
        x (np.array): Array of integers representing a grouping
        threshold (int): Minimum group size to keep.

    Returns:
        A np.array of integers but with any small groups set to 0.
    """
    x = x.copy()
    counts = x.value_counts()
    groups_to_remove = counts[counts < threshold].index
    x.loc[x.isin(groups_to_remove.values)] = 0
    # Reorder the categories removing the unused ones
    x_cat = pd.Categorical(x, categories=sorted(x.unique()))
    x_relabelled = pd.Series(x_cat.codes).values
    return x_relabelled


def identify_cal_amb_groups(
    hk,
    ZA_MFC_low=10,
    ZA_MFC_high=400,
    min_samples_in_group=1,
    max_time_between_cal_ambient=5,
):
    """
    Identifies consecutive pairs of calibration and ambient sampling times from 
    house keeping data.

    Args:
        - hk (pd.DataFrame): Housekeeping data as returned by load_hk.
        - ZA_MFC_low (float): The upper threshold for what is considered a cal 
            sample. Anything lower when task > 0 is considered a calibration.
        - ZA_MFC_high (float): The lower threshold for what is considered an 
            ambient sample. Anything higher when task > 0 is considered ambient.
        - min_samples_in_group (int): The minimum number of samples that make up
            a cal or ambient period.
        - max_time_between_cal_ambient (float): The maximum time in seconds 
            between a consecutive calibration and ambient point.

    Returns:
        A pandas DataFrame with 1 row per cal/ambient pairing with columns:
            - "group_number": Integer id
            - "cal_start": Datetime when the cal period started
            - "cal_end": Datetime when the cal period ended
            - "amb_start": Datetime when the amb period started
            - "amb_end": Datetime when the amb period ended
    """
    hk = hk.copy()
    ##########################################
    # Identifying cal and ambient timepoints
    ##########################################
    # Firstly create separate flags for whether a timepoint is a cal or ambient
    # (actual measurements will have False for both of these)
    hk["is_cal"] = (hk["Task"] > 0) & (hk["ZA_SB_MFC_Read"] < ZA_MFC_low) & (hk["Inlet_ZA_MFC_Read"] < 0.5)
    hk["is_ambient"] = (hk["Task"] > 0) & (hk["ZA_SB_MFC_Read"] > ZA_MFC_high) & (hk["Inlet_ZA_MFC_Read"] < 0.5)
    # Create a third column that species every timepoint as either cal, ambient, or measurement
    hk["group"] = np.select(
        [hk["is_cal"] == True, hk["is_ambient"] == True],
        ["cal", "ambient"],
        default="measurement",
    )

    ##########################################
    # Identifying cal and ambient groups
    ##########################################
    # Apply RLE to obtain cal and ambient groups
    hk["cal_group"] = rle(hk["is_cal"])
    hk["amb_group"] = rle(hk["is_ambient"])

    # Remove any groups that are too small
    hk["cal_group"] = remove_small_groups(hk["cal_group"], min_samples_in_group)
    hk["amb_group"] = remove_small_groups(hk["amb_group"], min_samples_in_group)

    ##########################################
    # Extracting time points of successive
    # cal and ambient groups
    ##########################################
    # First extract the start and end points of all the groups
    cal_timerange = (
        hk.loc[hk["cal_group"] != 0]
        .reset_index()
        .groupby("cal_group")
        .agg(
            start=pd.NamedAgg(column="Date_time", aggfunc=min),
            end=pd.NamedAgg(column="Date_time", aggfunc=max),
        )
        .sort_values("start")
    )
    amb_timerange = (
        hk.loc[
            hk["amb_group"] != 0,
        ]
        .reset_index()
        .groupby("amb_group")
        .agg(
            start=pd.NamedAgg(column="Date_time", aggfunc=min),
            end=pd.NamedAgg(column="Date_time", aggfunc=max),
        )
        .sort_values("start")
    )

    results = pd.DataFrame(
        columns=["group_number", "amb1_start", "amb1_end", "cal_start", "cal_end", "amb2_start", "amb2_end"]
    )
    
    ###CHANGED CODE
    
    # Manually remove points from cal_timerange and amb_timerange
    # Not sure how else to do this?
    cal_timerange_mod = cal_timerange[7:]
    #cal_timerange_mod = pd.concat(cal_timerange_mod[:])
    amb_timerange_mod = amb_timerange[4:67]
    #amb_timerange_mod = pd.concat(amb_timerange_mod[:])
    
    # Identify amb1 and amb2 groups before and after the cal group respectively
    for i, row in enumerate(cal_timerange_mod.itertuples()):
        amb1_group = (
            amb_timerange.loc[amb_timerange.iloc[::-1]["end"] <= row.start]
            .sort_values("start")
            .iloc[-1]
        )
        amb1_start = amb1_group["start"]
        amb1_end = amb1_group["end"]
        
        amb2_group = (
            amb_timerange.loc[amb_timerange["start"] >= row.end]
            .sort_values("start")
            .iloc[0]
        )
        amb2_start = amb2_group["start"]
        amb2_end = amb2_group["end"]
        
        #if i==0:
            #amb_len = (amb2_end - amb2_start)
            #amb1_start = row.start - amb_len
            
            # Find time immediately before cal start = this is amb1_end
            # Find time amb_length before amb1_end = this is amb1_start.
            # amb1_length is the time between an ambient start and end group
            #pass
        #else:
            #pass
            # Find all ambient groups that start BEFORE this cal group and 
            # then find the LATEST one

    # Allow a buffer time where ambient doesn't start immediately after cals
    #for row in cal_timerange.itertuples():
        # Find all ambient groups that start after this cal group and then
        # find the earliest one
        #amb_group = (
            #amb_timerange.loc[amb_timerange["start"] >= row.end]
            #.sort_values("start")
            #.iloc[0]
        #)

        if (amb1_end - amb1_start).total_seconds() < pd.Timedelta(10, 's').total_seconds():
            amb1_start = amb1_start - pd.Timedelta(23, 's')
        
        # See if this amb group falls within the tolerance range
        time_between_ambient_cal = (amb1_end - row.start).seconds
        time_between_cal_ambient = (amb2_start - row.end).seconds
        if time_between_ambient_cal and time_between_cal_ambient <= max_time_between_cal_ambient:
            row_df = pd.DataFrame(
                {
                    "group_number": len(results) + 1,
                    "amb1_start": amb1_start,
                    "amb1_end": amb1_end,
                    "cal_start": row.start,
                    "cal_end": row.end,
                    "amb2_start": amb2_start,
                    "amb2_end": amb2_end,
                },
                index=[len(results) + 1],
            )
            results = results.append(row_df)

    return results


def identify_cal_za_groups(
    hk,
    Inlet_MFC_high=2,
    min_samples_in_group=50,
):
    
    hk = hk.copy()
    
    hk["za_cal"] = (hk["Task"] > 0) & (hk["Inlet_ZA_MFC_Read"] > Inlet_MFC_high)
    hk["za_cal_group"] = rle(hk["za_cal"])
    hk["za_cal_group"] = remove_small_groups(hk["za_cal_group"], min_samples_in_group)
    
    za_cal_timerange = (
        hk.loc[hk["za_cal_group"] != 0]
        .reset_index()
        .groupby("za_cal_group")
        .agg(
            start=pd.NamedAgg(column="Date_time", aggfunc=min),
            end=pd.NamedAgg(column="Date_time", aggfunc=max),
        )
        .sort_values("start")
    )
    
    results = pd.DataFrame(
        columns=["group_number", "cal_start", "cal_end"]
    )
    
    model = "rbf"
    
    cal_start = []
    cal_end = []
    
    
    for i, row in enumerate (za_cal_timerange.itertuples()):
        algo = ruptures.Pelt(model=model, min_size=5, jump=5).fit(hk['SO2_Cal_MFC_Set_P'].loc[row.start:row.end].values)
        breakpoints = algo.predict(pen=3)
        breakpoints = breakpoints[:len(breakpoints)-1]
        breakpoint_times = hk.loc[row.start:row.end].index.values[breakpoints]
        
        cal_start.append(za_cal_timerange['start'][i+1])   
        
        for j in range(0, len(breakpoint_times - 1)):
            cal_end.append(breakpoint_times[j])
            cal_start.append(breakpoint_times[j])
            
        cal_end.append(za_cal_timerange['end'][i+1])
            
    results = pd.DataFrame({"group_number": len(results),
                            "cal_start": cal_start,
                            "cal_end": cal_end})
            
    return results


def calculate_amb_cal_points(hk, counts, cal_times, cyl_conc=5030, cal_remove_start=0, cal_remove_end=0, amb_remove_start=0, amb_remove_end=0):
    """
    Calculates the calibration parameters at each calibration/ambient grouping.

    Args:
        - hk (pd.DataFrame): The housekeeping data as returned by load_hk.
        - counts (pd.DataFrame): The binary counts data with the SO2 mixing 
            ratio calculated, as returned by load_counts.
        - cal_times (pd.DataFrame): Calibration and ambient period times as 
            returned by identify_cal_amb_groups.
        - mr_error (float): Value to multiply by mean mixing ratio to get 
            mixing ratio error.

    Returns:
        A pandas DataFrame containing the calibration parameters for each 
        calibration/ambient group pair.
        Has columns:
            - "time": The datetime the calibration period started
            - "group_number": The calibration/ambient group identifier
            - "mixing_ratio": The mean mixing ratio from the house keeping data
            - "mixing_ratio_error": The mixing ratio error, calculated as 
                mr_error * mixing_ratio.
            - "count": The mean cal counts minus the mean ambient counts.
            - "count_error": The error on the counts, derived as the standard 
                deviation of the calibration period counts.
    """
    # For each group find the Mixing Ratio, Counts, and associated errors
    cal_points = pd.DataFrame(
        columns=[
            "group_number",
            "task",
            "mixing_ratio",
            "mixing_ratio_error",
            "count",
            "count_error",
        ]
    )
    
    cal_so2_mfc = hk["Cal_SO2_MFC_Read"]
    cell_flow = hk["Cell_Flow"] * 1000
    
    ## cal ambient
    dy_da = []
    dy_db = []
    dy_dc = []


    #return counts, cal_times, remove_start, remove_end
    for i, row in enumerate(cal_times.itertuples()):
        mr = (
                hk.loc[row.cal_start : row.cal_end][int(cal_remove_start/2):int(-cal_remove_end/2)]
                .agg(mean=pd.NamedAgg(column="SO2_mr", aggfunc=np.mean))
                .squeeze()
            )
        task = (
                    hk.loc[row.cal_start : row.cal_end]
                    .agg(mean=pd.NamedAgg(column="Task", aggfunc=np.mean))
                    .squeeze()
                )
            #plt.plot(hk.loc[row.cal_start : row.cal_end][50:-50])
        counts_cal = (
                counts.loc[row.cal_start : row.cal_end][cal_remove_start:-cal_remove_end]
                .agg(
                    mean=pd.NamedAgg(column="Cts_diff", aggfunc=np.mean),
                    sd=pd.NamedAgg(column="Cts_diff", aggfunc=np.std),
                )
                .squeeze()
            )
        plt.plot(counts.loc[row.cal_start : row.cal_end][cal_remove_start:-cal_remove_end])
        counts_amb1 = (
                counts.loc[row.amb1_start : row.amb1_end][amb_remove_start:-amb_remove_end]
                .agg(
                    mean=pd.NamedAgg(column="Cts_diff", aggfunc=np.mean),
                    sd=pd.NamedAgg(column="Cts_diff", aggfunc=np.std),
                )
                .squeeze()
            )
        plt.plot(counts.loc[row.amb1_start : row.amb1_end][amb_remove_start:-amb_remove_end])
        counts_amb2 = (
                counts.loc[row.amb2_start : row.amb2_end][amb_remove_start:-amb_remove_end]
                .agg(
                    mean=pd.NamedAgg(column="Cts_diff", aggfunc=np.mean),
                    sd=pd.NamedAgg(column="Cts_diff", aggfunc=np.std),
                )
                .squeeze()
            )
        plt.plot(counts.loc[row.amb2_start : row.amb2_end][amb_remove_start:-amb_remove_end])
            # Determine how many points to remove at the end of the amb and cal groups
            #plt.plot(counts.loc[row.amb1_start : row.amb1_end][amb_remove:-5])
            #plt.plot(counts.loc[row.cal_start : row.cal_end][cal_remove:-5])
            #plt.plot(counts.loc[row.amb2_start : row.amb2_end][amb_remove:-5])
            
            # Determine whether to use the averaged amb group before or after the cal point or both
            #print(np.mean(counts_amb1), np.std(counts_amb1))
            #print(np.mean(counts_amb2), np.std(counts_amb2))
        counts_both = (np.mean([counts_amb1, counts_amb2]))
            #print(counts_both)
            #print(counts_cal["mean"])
            
        counts_out = (counts_cal.loc["mean"] - counts_both).squeeze()
        
        dy_da.append(cyl_conc / np.mean(cell_flow[row.cal_start:row.cal_end][int(cal_remove_start/2):int(-cal_remove_end/2)]))
        
        dy_db.append(np.mean(cal_so2_mfc[row.cal_start:row.cal_end][int(cal_remove_start/2):int(-cal_remove_end/2)]) / np.mean(cell_flow[row.cal_start:row.cal_end][int(cal_remove_start/2):int(-cal_remove_end/2)]))
        
        dy_dc.append(-(np.mean(cal_so2_mfc[row.cal_start:row.cal_end][int(cal_remove_start/2):int(-cal_remove_end/2)]) * cyl_conc) / np.mean(cell_flow[row.cal_start:row.cal_end][int(cal_remove_start/2):int(-cal_remove_end/2)])**2)
        
        mr_error = np.sqrt(((np.array(dy_da) * ((np.mean(cal_so2_mfc[row.cal_start:row.cal_end][int(cal_remove_start/2):int(-cal_remove_end/2)]) * 0.008) + 0.01))**2) + 
                           ((np.array(dy_db) * (cyl_conc * 0.05))**2) + ((np.array(dy_dc) * (np.mean(cell_flow[row.cal_start:row.cal_end][int(cal_remove_start/2):int(-cal_remove_end/2)]) * 0.03))**2))  
    
        count_error = counts_cal.loc["sd"].squeeze() / np.sqrt(len(counts.loc[row.cal_start : row.cal_end][cal_remove_start:-cal_remove_end]))
        
            # Save all results from this group into a DF
        group_results = pd.DataFrame(
                {
                    "time": row.cal_start,
                    "group_number": row.group_number,
                    "task": task,
                    "mixing_ratio": mr,
                    "mixing_ratio_error": mr_error[i]*1000,
                    "count": counts_out,
                    "count_error": count_error,
                },
                index=[i],
            )
    
        cal_points = cal_points.append(group_results)

    return cal_points


def calculate_za_cal_points(hk, counts, cal_times, cyl_conc=5030, remove_start=0, remove_end=0):
    
    # For each group find the Mixing Ratio, Counts, and associated errors
    cal_points = pd.DataFrame(
        columns=[
            "group_number",
            "task",
            "mixing_ratio",
            "mixing_ratio_error",
            "mixing_ratio_za_inlet_error",
            "count",
            "count_error",
        ]
    )

    cal_so2_mfc = hk["Cal_SO2_MFC_Read"]
    cell_flow = hk["Cell_Flow"] * 1000
    
    ## cal ambient
    dy_da = []
    dy_db = []
    dy_dc = []

    for i, row in enumerate(cal_times[5:].itertuples()):
        mr = (
                hk.loc[row.cal_start : row.cal_end][int(remove_start/2):int(-remove_end/2)]
                .agg(mean=pd.NamedAgg(column="SO2_mr", aggfunc=np.mean))
                .squeeze()
            )
        task = (
                hk.loc[row.cal_start : row.cal_end]
                .agg(mean=pd.NamedAgg(column="Task", aggfunc=np.mean))
                .squeeze()
            )
        counts_cal = (
                counts.loc[row.cal_start : row.cal_end][remove_start:-remove_end]
                .agg(
                    mean=pd.NamedAgg(column="Cts_diff", aggfunc=np.mean),
                    sd=pd.NamedAgg(column="Cts_diff", aggfunc=np.std),
                )
                .squeeze()
            )
        plt.plot(counts.loc[row.cal_start : row.cal_end][remove_start:-remove_end])
            # Determine how many points to remove at the end of the amb and cal groups
            #plt.plot(counts.loc[row.amb1_start : row.amb1_end][amb_remove:-5])
            #plt.plot(counts.loc[row.cal_start : row.cal_end][cal_remove:-5])
            #plt.plot(counts.loc[row.amb2_start : row.amb2_end][amb_remove:-5])
            
        dy_da.append(cyl_conc / np.mean(cell_flow[row.cal_start:row.cal_end][int(remove_start/2):int(-remove_end/2)]))
        
        dy_db.append(np.mean(cal_so2_mfc[row.cal_start:row.cal_end][int(remove_start/2):int(-remove_end/2)]) / np.mean(cell_flow[row.cal_start:row.cal_end][int(remove_start/2):int(-remove_end/2)]))
        
        dy_dc.append(-(np.mean(cal_so2_mfc[row.cal_start:row.cal_end][int(remove_start/2):int(-remove_end/2)]) * cyl_conc) / np.mean(cell_flow[row.cal_start:row.cal_end][int(remove_start/2):int(-remove_end/2)])**2)
        
        mr_error = np.sqrt(((np.array(dy_da) * ((np.mean(cal_so2_mfc[row.cal_start:row.cal_end][int(remove_start/2):int(-remove_end/2)]) * 0.008) + 0.01))**2) + 
                           ((np.array(dy_db) * (cyl_conc * 0.05))**2) + ((np.array(dy_dc) * (np.mean(cell_flow[row.cal_start:row.cal_end][int(remove_start/2):int(-remove_end/2)]) * 0.03))**2))  
    
        count_error = counts_cal.loc["sd"].squeeze() / np.sqrt(len(counts.loc[row.cal_start : row.cal_end][remove_start:-remove_end]))
    
            # Save all results from this group into a DF
        group_results = pd.DataFrame(
                {
                    "time": row.cal_start,
                    "group_number": row.group_number,
                    "task": task,
                    "mixing_ratio": mr,
                    "mixing_ratio_error": mr_error[i]*1000,
                    "count": counts_cal.loc["mean"].squeeze(),
                    "count_error": count_error,
                },
                index=[i],
            )
    
        cal_points = cal_points.append(group_results)

    return cal_points


def plot_cal_groups(hk, groups, start=None, end=None):
    """
    Plots the calibration and ambient groupings on a time-series of the 
      housekeeping data.

    Args:
        - hk (pd.DataFrame): The housekeeping data as returned by load_hk.
        - groups (pd.DataFrame): The calibration and ambient groups as returned
            by identify_cal_amb_groups.
        - start (Datetime, optional): Earliest time to plot, if not provided 
            then plots from the earliest point available.
        - end (Datetime, optional): Latest time to plot, if not provided then 
            plots until the latest point available.

    Returns:
        None, plots instead.
    """
    hk = hk.copy()

    # Because pandas doesn't have an interval join we need to do a
    # 2-step join, firstly on the lower bound followed by one on the upper bound
    # Do this for both cals and ambient to group points into sample/cal/ambient
    # Cals
    full = pd.merge_asof(
        hk,
        groups[["group_number", "cal_start", "cal_end"]].rename(
            columns={"group_number": "group_cal"}
        ),
        left_index=True,
        right_on="cal_start",
        direction="backward",  # Lower-bound
    )
    full = pd.merge_asof(
        full,
        groups[["group_number", "cal_start", "cal_end"]].rename(
            columns={"group_number": "group_cal"}
        ),
        left_index=True,
        right_on="cal_end",
        direction="forward",  # Upper-bound
    )
    # Ambient
    full = pd.merge_asof(
        full,
        groups[["group_number", "amb1_start", "amb1_end"]].rename(
            columns={"group_number": "group_amb"}
        ),
        left_index=True,
        right_on="amb1_start",
        direction="backward",  # Lower-bound
    )
    full = pd.merge_asof(
        full,
        groups[["group_number", "amb1_start", "amb1_end"]].rename(
            columns={"group_number": "group_amb"}
        ),
        left_index=True,
        right_on="amb1_end",
        direction="forward",  # Upper-bound
    )

    # Overlaping intervals have the two groups (one from each join) having the
    # same value. Use this to create a column that is either measurement/amb/cal
    full["group"] = np.select(
        [
            full["group_cal_x"] == full["group_cal_y"],
            full["group_amb_x"] == full["group_amb_y"],
        ],
        ["cal", "ambient"],
        default="measurement",
    )

    # Finally, create column with definitive group number
    full["group_number"] = np.select(
        [full["group"] == "cal", full["group"] == "ambient"],
        [full["group_cal_x"], full["group_amb_x"]],
        default=0,
    )

    if start is not None:
        full = full.loc[start:]

    if end is not None:
        full = full.loc[:end]

    figure, axes = plt.subplots()
    sns.scatterplot(
        x="Date_time",
        y="Signal_LIF_Counts",
        hue="group_number",
        data=full.reset_index(),
        ax=axes,
        style="group",
        palette="Dark2",
        linewidth=0.3,
    )
    axes.legend(loc="upper left", bbox_to_anchor=(0, 1))
    plt.show()


def plot_cals(cal_points, Ri=0, b0=30, eps=1e-7):
    """
    Plots a scatter of mixing ratio against counts to obtain the instrument 
    sensitivity.

    Args:
        - cal_points (pd.DataFrame): The calibration parameters as calculated 
            by calculate_cal_points.

    Returns:
        None, plots instead.
    """
    
    x = np.array(cal_points['mixing_ratio'])
    y = np.array(cal_points['count'])
    sx = np.array(cal_points['mixing_ratio_error'])
    sy = np.array(cal_points['count_error'])
    
    #tol = 1e-7
    tol = abs(b0) * eps # the fit will stop iterating when the slope converges to within this value
    
    wx = 1 / (sx**2) # weight x
    wy = 1 / (sy**2) # weight y
    
    # iterative calculation of slope and intercept
    b = b0
    b_diff = tol + 1
    while(b_diff > tol):
        b_old = b
        alpha_i = np.sqrt(wx*wy)
        Wi = (wx*wy) / ((b**2)*wy + wx - 2*b*Ri*alpha_i)
        WiX = Wi*x
        WiY = Wi*y
        WiX = WiX[~np.isnan(WiX)]
        WiY = WiY[~np.isnan(WiY)]
        Wi = Wi[~np.isnan(Wi)]
        sumWiX = sum(WiX)
        sumWiY = sum(WiY)
        sumWi = sum(Wi)
        Xbar = sumWiX / sumWi
        Ybar = sumWiY / sumWi
        Ui = x - Xbar
        Vi = y - Ybar
        
        Bi = Wi * ((Ui/wy) + (b*Vi/wx) - (b*Ui+Vi)*Ri/alpha_i)
        wTOPint = Bi*Wi*Vi
        wBOTint = Bi*Wi*Ui
        wTOPint = wTOPint[~np.isnan(wTOPint)]
        wBOTint = wBOTint[~np.isnan(wBOTint)]
        sumTOP = sum(wTOPint)
        sumBOT = sum(wBOTint)
        b = sumTOP / sumBOT
        
        b_diff = abs(b - b_old)
        
        a = Ybar - b*Xbar

    # error calculation
    Xadj = Xbar + Bi
    WiXadj = Wi * Xadj
    WiXadj = WiXadj[~np.isnan(WiXadj)]
    sumWiXadj = sum(WiXadj)
    Xadjbar = sumWiXadj / sumWi
    Uadj = Xadj - Xadjbar
    wErrorTerm = Wi * Uadj * Uadj
    wErrorTerm = wErrorTerm[~np.isnan(wErrorTerm)]
    errorSum = sum(wErrorTerm)
    b_err = np.sqrt(1 / errorSum)
    a_err = np.sqrt((1/sumWi) + (Xadjbar**2)*(b_err**2))
    
    # goodness of fit calculation
    lgth = len(x)
    wSint = Wi * (y - b*x - a)**2
    wSint = wSint[~np.isnan(wSint)]
    sumSint = sum(wSint)
    
    # plot trendline
    fig, ax = plt.subplots()
    ax.errorbar(x, y, sy, sx, marker='o', linestyle='')
    ax.set_xlabel('SO$_2$ Mixing Ratio (ppt)', fontsize=14)
    ax.set_ylabel('LIF Signal (cps/mW)', fontsize=14)
    yp = b*x + a
    ax.plot(x, yp, linestyle='dashed')
    ax.tick_params(axis='both', labelsize=12)
    ax.annotate('sensitivity = %.1f +/- %.1f\nintercept = %.0f +/- %.0f' % (b, b_err, a, a_err), xycoords='axes fraction', xy=(0.05,0.9), fontsize=12)
    
    # return mean cal time, slope, error in slope, intercept, error in intercept and goodness of fit values
    time_avg = np.mean(cal_points["time"])
    return(time_avg, b, b_err, a, a_err, (sumSint/(lgth-2)), np.sqrt(2/(lgth-2)))   

    
def plot_cals_za_inlet(cal_points):
    """
    Plots a scatter of mixing ratio against counts to obtain the instrument 
    sensitivity.

    Args:
        - cal_points (pd.DataFrame): The calibration parameters as calculated 
            by calculate_cal_points.

    Returns:
        None, plots instead.
    """
    # calculate overall sensitivity error
    # NB: I'm not sure I follow what this is doing but I'm happy to use it!
    
    overall_error_below = np.sqrt(
        cal_points["mixing_ratio_za_inlet_error"] ** 2 + cal_points["count_error"] ** 2
    )
    # plot trendline, calculate slope which is equal to sensitivity and print 
    # sensitivity on plot
    slope, error = np.polyfit(
        cal_points["mixing_ratio_za_inlet"],
        cal_points["count"],
        1,
        cov=True,
        w=overall_error_below,
    )
    sens_error = np.sqrt(np.diag(error))[0]

    # Plot calibration points
    figure, axes_scatter = plt.subplots()

    # Scatter plot
    axes_scatter.errorbar(
        cal_points["mixing_ratio_za_inlet"],
        cal_points["count"],
        yerr=cal_points["count_error"],
        xerr=cal_points["mixing_ratio_za_inlet_error"],
        ls="none",
        marker="o",
        color="red",
    )

    # Plot linear regression line
    sns.regplot(x="mixing_ratio_za_inlet", y="count", data=cal_points, ax=axes_scatter)

    axes_scatter.set_xlabel("SO2 mixing ratio (ppt)")
    axes_scatter.set_ylabel(
        "Online-offline normalised signal LIF counts (cps/V)"
    )
    axes_scatter.annotate(
        "sensitivity = %.2f \u00B1 %.2f\nintercept = %.2f" % (slope[0], sens_error, slope[1]),
        xycoords="axes fraction",
        xy=(0.05, 0.95),
    )

    plt.show()
    
    time_avg = np.mean(cal_points["time"])
    
    print([time_avg, slope[0], sens_error, slope[1]])
