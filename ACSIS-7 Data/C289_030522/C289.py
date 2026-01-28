import C289_functions as lif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
import datetime
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import math


#### run data - sig counts
#from LIF_processes import reprocess_binary_data
#reprocess_binary_data(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\C289_030522\LIFCnts_20220503', 
                      #'03/05/2022 11:12:47', skip_start=1, n_bad_online=1)

#### run data - ref counts
#from LIF_processes_ref import reprocess_binary_data
#reprocess_binary_data(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\C289_030522\LIFCnts_20220503', 
                      #'03/05/2022 11:12:47', skip_start=1, n_bad_online=1)


# Load signal counts
sig_cts_data = lif.load_counts(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\C289_030522\processed data_030723_sig')
print(sig_cts_data.index[0])
print(sig_cts_data.index[-1])
sig_cts_diff = pd.DataFrame(sig_cts_data['SO2_pptv'])
sig_cts_diff = sig_cts_diff.rename(columns={'SO2_pptv':'Cts_diff'})

# Load ref counts
ref_cts_data = lif.load_counts(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\C289_030522\processed data_030723_ref')
print(ref_cts_data.index[0])
print(ref_cts_data.index[-1])
ref_cts_diff = pd.DataFrame(ref_cts_data['SO2_pptv'])
ref_cts_diff = ref_cts_diff.rename(columns={'SO2_pptv':'Cts_diff'})


# Plot sensitivity with time
sens_amb = pd.read_excel('Sensitivities_amb_030723.xlsx')
sens_amb['Average Time'] = pd.to_datetime(sens_amb['Average Time'], format='%Y-%m-%d %H:%M:%S.%f')

sens_za = pd.read_excel('Sensitivities_za_030723.xlsx')
sens_za['Average Time'] = pd.to_datetime(sens_za['Average Time'], format='%Y-%m-%d %H:%M:%S.%f')

fig, ax = plt.subplots()
ax.errorbar(x=sens_amb['Average Time'], y=sens_amb['Sensitivity'], yerr=sens_amb['Sensitivity Error'], 
             color='blue', marker='o', ms=8, ls='')
ax.errorbar(x=sens_za['Average Time'], y=sens_za['Sensitivity'], yerr=sens_za['Sensitivity Error'],
            color='orange', marker='o', ms=8, ls='')
ax.legend(['ambient', 'za'], fontsize=16)
ax.set_xlabel('Date Time (UTC)', fontsize=18)
ax.set_ylabel('Sensitivity (cps/mW/ppt)', fontsize=18)
ax.axes.tick_params(axis='both', labelsize=16)


# Read in FAAM altitude and SO2 data
FAAM_data = Dataset(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\FAAM_data\core_faam_20220503_v005_r1_c289_1hz.nc', format='NETCDF4_CLASSIC')
nctime = FAAM_data.variables['Time'][:]
t_unit = FAAM_data.variables['Time'].units
t_cal = FAAM_data.variables['Time'].calendar
tvalue = num2date(nctime, units=t_unit, calendar=t_cal)
str_time = [i.strftime("%Y-%m-%d %H:%M:%S") for i in tvalue]

ncalt = FAAM_data.variables['ALT_GIN'][:]
Alt_df = pd.DataFrame(ncalt, str_time)
Alt_df.index = pd.to_datetime(Alt_df.index, utc=True)
plt.plot(Alt_df)

ncTECO = FAAM_data.variables['SO2_TECO'][:]
TECO_df = pd.DataFrame(ncTECO, str_time, columns=['SO2_ppb'])
TECO_df.index = pd.to_datetime(TECO_df.index, utc=True)
TECO_df.index = TECO_df.index.rename('Date_time')
TECO_df = TECO_df.sort_values(by="Date_time")
FAAM_SO2_mr_1hz = TECO_df.loc['2022-05-03 14:44:00':'2022-05-03 16:05:00']


# Plot sensitivity with altitude
fig, ax = plt.subplots()
ax.errorbar(x=sens_amb['Average Time'], y=sens_amb['Sensitivity'], yerr=sens_amb['Sensitivity Error'], 
             color='blue', marker='o', ms=8, ls='')
ax.errorbar(x=sens_za['Average Time'], y=sens_za['Sensitivity'], yerr=sens_za['Sensitivity Error'],
            color='orange', marker='o', ms=8, ls='')
ax.legend(['ambient', 'za'], fontsize=16)
ax.set_xlabel('Date Time (UTC)', fontsize=18)
ax.set_ylabel('Sensitivity (cps/mW/ppt)', fontsize=18)
ax.axes.tick_params(axis='both', labelsize=16)

ax2 = ax.twinx()
ax2.plot(Alt_df, color='grey')
#ax2.set_xticks(str_time[::2500])
ax2.set_ylabel('Altitude (m)', fontsize=18, color='grey')
ax2.axes.tick_params(axis='both', labelsize=16)


# Calculate mean sensitivity
sens_all = pd.read_excel('All_sensitivities_030723.xlsx')
sens_all['Average Time'] = pd.to_datetime(sens_all['Average Time'], format='%Y-%m-%d %H:%M:%S.%f')
sens_mean = np.mean(sens_all['Sensitivity'])
sens_mean_err = np.mean(sens_all['Sensitivity Error'])

SO2_mr = sig_cts_diff / sens_mean

SO2_mr_filt = SO2_mr.loc['2022-05-03 14:44:00':'2022-05-03 16:05:00']
plt.plot(SO2_mr_filt)

cal_times_all = pd.read_excel('All_cal_times.xlsx')
cal_times_all['Cal_start'] = pd.to_datetime(cal_times_all['Cal_start'], format='%Y-%m-%d %H:%M:%S.%f')
cal_times_all['Cal_end'] = pd.to_datetime(cal_times_all['Cal_end'], format='%Y-%m-%d %H:%M:%S.%f')

for i in range(len(cal_times_all)):
    SO2_mr_filt['Cts_diff'].mask((SO2_mr_filt.index > str(cal_times_all['Cal_start'][i])) &
                       (SO2_mr_filt.index < str(cal_times_all['Cal_end'][i])), np.nan, inplace=True)
    #Cts_diff_filt['Flag'].mask((Cts_diff_filt.index > str(cal_times_all['Amb_start'][i])) &
                       #(Cts_diff_filt.index < str(cal_times_all['Amb_end'][i])), 1, inplace=True)

plt.plot(SO2_mr_filt, color='orange')

fig, ax = plt.subplots()
ax.plot(SO2_mr_filt['Cts_diff'], color='orange')
ax2 = ax.twinx()
ax2.plot(Alt_df)
ax2.set_ylabel('Altitude (m)', fontsize=18)
ax2.axes.tick_params(axis='both', labelsize=16)

# save data at 10 Hz
#SO2_mr_filt = SO2_mr_filt / 1000
SO2_mr_filt = SO2_mr_filt.rename(columns={'Cts_diff':'SO2_ppt'})
#SO2_mr_filt = SO2_mr_filt.loc['2022-05-02 11:20:00':'2022-05-02 16:30:00']
#SO2_mr_10hz = SO2_mr_filt
#plt.plot(SO2_mr_10hz)
#SO2_mr_10hz.to_csv('LIF_SO2_mr_10Hz_C287.csv')
plt.plot(SO2_mr_filt)

## average data to 10 s and save data
SO2_mr_10s = SO2_mr_filt.resample('10s').mean()
SO2_mr_10s.to_csv('York_LIF_SO2_10s_C289.csv')

plt.plot(SO2_mr_10s)

