import C285_functions as lif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
import datetime
import math
from time_stamps import time_average
import scipy

#### run data - sig counts
#lif.reprocess_binary_data_sig(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\LIFCnts_20220430', 
                      #'30/04/2022 12:11:57', skip_start=1, n_bad_online=1)

#### run data - ref counts
#lif.reprocess_binary_data_ref(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\LIFCnts_20220430', 
                      #'30/04/2022 12:11:57', skip_start=1, n_bad_online=1)


# Load signal counts
sig_cts_data = lif.load_counts(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\processed data_050224_sig')
print(sig_cts_data.index[0])
print(sig_cts_data.index[-1])
sig_cts_diff = pd.DataFrame(sig_cts_data['Cts_diff'])
plt.plot(sig_cts_diff)

# Load ref counts
ref_cts_data = lif.load_counts(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\processed data_050224_ref')
print(ref_cts_data.index[0])
print(ref_cts_data.index[-1])
ref_cts_diff = pd.DataFrame(ref_cts_data['Cts_diff'])
plt.plot(ref_cts_diff)


# Plot sensitivity with time
sens_amb = pd.read_excel('Sensitivities_amb_030924.xlsx')
sens_amb['Average Time'] = pd.to_datetime(sens_amb['c285_amb_time'], format='%Y-%m-%d %H:%M:%S.%f')
sens_amb_mean = np.mean(sens_amb['c285_amb_sens'])
sens_amb_std = np.std(sens_amb['c285_amb_sens'])

sens_za = pd.read_excel('Sensitivities_za_030924.xlsx')
sens_za['Average Time'] = pd.to_datetime(sens_za['c285_za_time'], format='%Y-%m-%d %H:%M:%S.%f')
sens_za_mean = np.mean(sens_za['c285_za_sens'])
sens_za_std = np.std(sens_za['c285_za_sens'])

fig, ax = plt.subplots()
ax.errorbar(x=sens_amb['Average Time'], y=sens_amb['c285_amb_sens'], yerr=sens_amb['Sensitivity Error'], 
             color='blue', marker='o', ms=8, ls='')
ax.errorbar(x=sens_za['Average Time'], y=sens_za['c285_za_sens'], yerr=sens_za['Sensitivity Error'],
            color='orange', marker='o', ms=8, ls='')
ax.legend(['ambient', 'za'], fontsize=16)
ax.set_xlabel('Date Time (UTC)', fontsize=18)
ax.set_ylabel('Sensitivity (cps/mW/ppt)', fontsize=18)
ax.axes.tick_params(axis='both', labelsize=16)


# Read in FAAM altitude and SO2 data
FAAM_data = Dataset(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\FAAM_data\core_faam_20220430_v005_r2_c285_1hz.nc', format='NETCDF4_CLASSIC')
nctime = FAAM_data.variables['Time'][:]
t_unit = FAAM_data.variables['Time'].units
t_cal = FAAM_data.variables['Time'].calendar
tvalue = num2date(nctime, units=t_unit, calendar=t_cal)
str_time = [i.strftime("%Y-%m-%d %H:%M:%S") for i in tvalue]

ncalt = FAAM_data.variables['ALT_GIN'][:]
Alt_df = pd.DataFrame(ncalt, str_time, columns=['Altitude'])
Alt_df.index = pd.to_datetime(Alt_df.index, utc=True)
Alt_df.index = Alt_df.index.rename('Date_time')

ncTECO = FAAM_data.variables['SO2_TECO'][:]
TECO_df = pd.DataFrame(ncTECO, str_time, columns=['SO2_ppb'])
TECO_df.index = pd.to_datetime(TECO_df.index, utc=True)
TECO_df.index = TECO_df.index.rename('Date_time')
TECO_df = TECO_df.sort_values(by="Date_time")
FAAM_SO2_mr_1hz = TECO_df.loc['2022-04-30 12:25:21':'2022-04-30 16:55:00']
#FAAM_SO2_mr_1hz = FAAM_SO2_mr_1hz.interpolate(method='linear')
#FAAM_SO2_mr_1hz.to_csv('Final_FAAM_SO2_mr_1Hz_C285_interp.csv')

plt.plot(Alt_df)
plt.plot(FAAM_SO2_mr_1hz)

# Plot sensitivity with altitude
fig, ax = plt.subplots()
ax.errorbar(x=sens_amb['Average Time'], y=sens_amb['c285_amb_sens'], yerr=sens_amb['Sensitivity Error'], 
             color='blue', marker='o', ms=8, ls='')
ax.errorbar(x=sens_za['Average Time'], y=sens_za['c285_za_sens'], yerr=sens_za['Sensitivity Error'],
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
sens_all = pd.read_excel('All_sensitivities_030924.xlsx')
sens_all['Average Time'] = pd.to_datetime(sens_all['Average Time'], format='%Y-%m-%d %H:%M:%S.%f')
sens_mean = np.mean(sens_all['Sensitivity'])
sens_mean_error = np.mean(sens_all['Sensitivity Error'])
sens_std = np.std(sens_all['Sensitivity'])

SO2_mr = sig_cts_diff / sens_mean


# Remove cal times
SO2_mr_filt = SO2_mr

cal_times_all = pd.read_excel('All_cal_times.xlsx')
cal_times_all['Amb_start'] = pd.to_datetime(cal_times_all['Amb_start'], format='%Y-%m-%d %H:%M:%S.%f')
cal_times_all['Amb_end'] = pd.to_datetime(cal_times_all['Amb_end'], format='%Y-%m-%d %H:%M:%S.%f')
cal_times_all['Cal_start'] = pd.to_datetime(cal_times_all['Cal_start'], format='%Y-%m-%d %H:%M:%S.%f')
cal_times_all['Cal_end'] = pd.to_datetime(cal_times_all['Cal_end'], format='%Y-%m-%d %H:%M:%S.%f')

for i in range(len(cal_times_all)):
    SO2_mr_filt['Cts_diff'].mask((SO2_mr_filt.index > str(cal_times_all['Cal_start'][i])) &
                       (SO2_mr_filt.index < str(cal_times_all['Cal_end'][i])), np.nan, inplace=True)
    #Cts_diff_filt['Flag'].mask((Cts_diff_filt.index > str(cal_times_all['Amb_start'][i])) &
                       #(Cts_diff_filt.index < str(cal_times_all['Amb_end'][i])), 1, inplace=True)

plt.plot(SO2_mr_filt['Cts_diff'], color='orange')

fig, ax = plt.subplots()
ax.plot(SO2_mr_filt['Cts_diff'], color='orange')
ax2 = ax.twinx()
ax2.plot(Alt_df.loc['2022-04-30 10:00:22':'2022-04-30 17:15:23'])
ax2.set_ylabel('Altitude (m)', fontsize=18)
ax2.axes.tick_params(axis='both', labelsize=16)

# Save data at 10 Hz
SO2_mr_filt = SO2_mr_filt / 1000
SO2_mr_filt = SO2_mr_filt.rename(columns={'Cts_diff':'SO2_ppb'})

# average to 4 Hz
SO2_mr_4hz = SO2_mr_filt.resample('250ms').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
SO2_mr_4hz = SO2_mr_4hz.loc['2022-04-30 12:25:21':'2022-04-30 16:55:00']

## average data to 1 s and save data
SO2_mr_5hz = SO2_mr_filt.resample('200ms').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
SO2_mr_5hz = SO2_mr_5hz.loc['2022-04-30 12:25:21':'2022-04-30 16:55:00']
#SO2_mr_5hz = SO2_mr_5hz.interpolate(method='linear')
#SO2_mr_5hz.to_csv('Final_LIF_SO2_mr_5Hz_C285.csv')
plt.plot(SO2_mr_5hz)

SO2_mr_1hz = SO2_mr_filt.resample('1s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
SO2_mr_1hz = SO2_mr_1hz.loc['2022-04-30 12:25:21':'2022-04-30 16:55:00']
#SO2_mr_1hz.to_csv('Final_LIF_SO2_mr_1Hz_C285.csv')

## average data to 10 s and save data
SO2_mr_10s = SO2_mr_filt.resample('10s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
SO2_mr_10s = SO2_mr_10s.loc['2022-04-30 12:25:21':'2022-04-30 16:55:00']
#SO2_mr_10s.to_csv('Final_LIF_SO2_mr_10s_C285.csv')


# Load CIMS SO2 4Hz CPS APR25 data
CIMS_SO2_mr = pd.read_csv('C:/Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\CIMS_data\ACRUISE3_CIMS_SO2_4HZ_CPS_APR25.csv')
CIMS_SO2_mr['Date_time'] = pd.to_datetime(CIMS_SO2_mr['t_start_Buf'], format="%Y/%m/%d %H:%M:%S", utc=True)
CIMS_SO2_mr = CIMS_SO2_mr.set_index('Date_time', drop=True)
CIMS_SO2_mr = CIMS_SO2_mr.sort_values(by="Date_time")

CIMS_SO2_mr = CIMS_SO2_mr.drop(['flight_id', 't_start_Buf'], axis=1)
CIMS_SO2_mr = CIMS_SO2_mr.rename(columns={'SO2_cps':'SO2_ppb'})
#CIMS_SO2_mr = CIMS_SO2_mr.resample('250ms').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
CIMS_SO2_mr_1hz = CIMS_SO2_mr.resample('1s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
CIMS_SO2_mr_1hz = CIMS_SO2_mr_1hz.loc['2022-04-30 12:25:21':'2022-04-30 16:55:00']
CIMS_SO2_mr_4hz = CIMS_SO2_mr_4hz / 11.966


# Load CO2 data
CO2_data = pd.read_csv(r"C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\CO2_data\faam-fgga_faam_20220430_r0_c285.na", header=54, sep=' ',
                       names=['Date_time', 'CO2_ppm', 'CO2_flag', 'Methane_ppb', 'Methane_flag', 'flow_status'])
Ref_date = pd.to_datetime('2022-04-30 00:00:00').tz_localize(None)
CO2_data['Date_time'] = pd.to_datetime(CO2_data['Date_time'], unit='s', origin=Ref_date)
CO2_data = CO2_data.set_index('Date_time', drop=True)
CO2_data = CO2_data.sort_values(by="Date_time")
CO2_data = CO2_data['CO2_ppm']
#CO2_data = CO2_data.resample('250ms').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
CO2_data = CO2_data.loc['2022-04-30 12:25:21':'2022-04-30 16:55:00']
CO2_data = pd.DataFrame(CO2_data)
CO2_data.replace(999.99, np.nan, inplace=True)
#CO2_data.dropna(inplace=True)
#CO2_data = CO2_data.interpolate(method='linear')
#CO2_data.to_csv('CO2_mr_4Hz_C285_interp.csv')


###############################################################################
# Comparison of response time for Fig. 9
CIMS_SO2_mr_4hz.index = CIMS_SO2_mr_4hz.index - pd.Timedelta(seconds=1.55)
FAAM_SO2_mr_1hz.index = FAAM_SO2_mr_1hz.index - pd.Timedelta(seconds=1.7)
CO2_data.index = CO2_data.index + pd.Timedelta(seconds=1.45)
fig, ax = plt.subplots()
ax.plot(SO2_mr_5hz, color='blue')
ax.plot(FAAM_SO2_mr_1hz, color='orange')
ax.plot(CIMS_SO2_mr_4hz, color='green')
ax2 = ax.twinx()
ax2.plot(CO2_data, color='red')
plt.xlabel('Date time (UTC)', fontsize=18)
plt.ylabel('SO2 mixing ratio (ppb)', fontsize=18)
plt.tick_params(axis='both', labelsize=16)
plt.legend(['LIF', 'PF', 'CIMS'], fontsize=16)

SO2_mr_5hz.index = SO2_mr_5hz.index.tz_localize(None)
#SO2_mr_5hz.to_excel('Final_LIF_SO2_mr_5Hz_rt.xlsx')

FAAM_SO2_mr_1hz.index = FAAM_SO2_mr_1hz.index.tz_localize(None)
#FAAM_SO2_mr_1hz.to_excel('Final_FAAM_SO2_mr_1Hz_rt.xlsx')

CIMS_SO2_mr_4hz.index = CIMS_SO2_mr_4hz.index.tz_localize(None)
#CIMS_SO2_mr_4hz.to_excel('Final_CIMS_SO2_mr_4Hz_rt.xlsx')

CO2_data.index = CO2_data.index.tz_localize(None)
#CO2_data.to_excel('Final_CO2_mr_10Hz_rt.xlsx')


###############################################################################
# Plume analysis - to be imported into script 'Plume_int_C285' for integration
# and regression analysis for Fig. 10 and similar SI plots
CO2_data_4hz = CO2_data.resample('250ms').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))

#CIMS_SO2_mr_4hz.index = CIMS_SO2_mr_4hz.index - pd.Timedelta(seconds=0.5)
CIMS_SO2_mr_4hz.index = CIMS_SO2_mr_4hz.index - pd.Timedelta(seconds=1.5)
CO2_data_4hz.index = CO2_data_4hz.index + pd.Timedelta(seconds=1.6)
FAAM_SO2_mr_1hz.index = FAAM_SO2_mr_1hz.index - pd.Timedelta(seconds=1.7)

CO2_data_4hz = CO2_data_4hz.interpolate(method='linear')
CIMS_SO2_mr_4hz = CIMS_SO2_mr_4hz.interpolate(method='linear')
SO2_mr_4hz = SO2_mr_4hz.interpolate(method='linear')
FAAM_SO2_mr_1hz = FAAM_SO2_mr_1hz.interpolate(method='linear')

fig, ax = plt.subplots()
ax.plot(CIMS_SO2_mr_4hz, color='green')
ax.plot(SO2_mr_4hz, color='blue')
ax.plot(FAAM_SO2_mr_1hz, color='orange')
ax2 = ax.twinx()
ax2.plot(CO2_data_4hz, color='red')

CO2_data_Picarro = CO2_data_4hz.loc['2022-04-30 14:25:20':'2022-04-30 14:34:44']
SO2_data_CIMS = CIMS_SO2_mr_4hz.loc['2022-04-30 14:25:20':'2022-04-30 14:34:44']
SO2_data_LIF = SO2_mr_4hz.loc['2022-04-30 14:25:20':'2022-04-30 14:34:44']
SO2_data_PF = FAAM_SO2_mr_1hz.loc['2022-04-30 14:25:20':'2022-04-30 14:34:44']

fig, ax = plt.subplots()
ax.plot(SO2_data_LIF, color='blue')
ax.plot(SO2_data_CIMS, color='green')
ax.plot(SO2_data_PF, color='orange')
ax2 = ax.twinx()
ax2.plot(CO2_data_Picarro, color='red')

#CO2_data_Picarro.to_csv('C285_CO2_tm_int_4hz.csv')
#SO2_data_CIMS.to_csv('C285_SO2_CIMS_tm_int_4hz_cps.csv')
#SO2_data_LIF.to_csv('C285_SO2_LIF_tm_int_4hz.csv')
#SO2_data_PF.to_csv('C285_SO2_PF_tm_int_1s.csv')


#################################### Plume 2
# CO2_data_4hz = CO2_data.resample('250ms').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))

# CIMS_SO2_mr_4hz.index = CIMS_SO2_mr_4hz.index - pd.Timedelta(seconds=0.5)
# CO2_data_4hz.index = CO2_data_4hz.index + pd.Timedelta(seconds=1.6)

# CO2_data_4hz = CO2_data_4hz.interpolate(method='linear')
# CIMS_SO2_mr_4hz = CIMS_SO2_mr_4hz.interpolate(method='linear')
# SO2_mr_4hz = SO2_mr_4hz.interpolate(method='linear')

# fig, ax = plt.subplots()
# ax.plot(SO2_mr_4hz, color='blue')
# ax.plot(CIMS_SO2_mr_4hz, color='green')
# ax2 = ax.twinx()
# ax2.plot(CO2_data_4hz, color='red')

# CO2_data_Picarro = CO2_data_4hz.loc['2022-04-30 13:35:15':'2022-04-30 13:43:30']
# SO2_data_CIMS = CIMS_SO2_mr_4hz.loc['2022-04-30 13:35:15':'2022-04-30 13:43:30']
# SO2_data_LIF = SO2_mr_4hz.loc['2022-04-30 13:35:15':'2022-04-30 13:43:30']

# fig, ax = plt.subplots()
# ax.plot(SO2_data_LIF, color='blue')
# ax.plot(SO2_data_CIMS, color='green')
# ax2 = ax.twinx()
# ax2.plot(CO2_data_Picarro, color='red')

# #CO2_data_Picarro.to_csv('C285_CO2_tm_int_2.csv')
# #SO2_data_CIMS.to_csv('C285_SO2_CIMS_tm_int_2.csv')
# #SO2_data_LIF.to_csv('C285_SO2_LIF_tm_int_2.csv')


###############################################################################
# 10 s time match analysis for all instruments for Figs. 7 and 11
# to be imported into script 'Correlation_concat' for Fig. 8
plt.plot(CIMS_SO2_mr_1hz, color='green', alpha=0.7)
plt.plot(FAAM_SO2_mr_1hz, color='orange', alpha=0.7)
plt.plot(SO2_mr_1hz, color='blue', alpha=0.6)

# time match data and remove nans
FAAM_mr_time_correct = []
LIF_mr_time_correct = []
CIMS_mr_time_correct = []
Time_correct = []

for t_FAAM, t_LIF, t_CIMS, FAAM_SO2, LIF_SO2, CIMS_SO2 in zip(FAAM_SO2_mr_1hz.index[4:], SO2_mr_1hz.index[:-3], CIMS_SO2_mr_1hz.index[:-1], FAAM_SO2_mr_1hz['SO2_ppb'][4:], SO2_mr_1hz['SO2_ppb'][:-3], CIMS_SO2_mr_1hz['SO2_ppb'][:-1]):
    if not math.isnan(LIF_SO2):
        if not math.isnan(FAAM_SO2):
            if not math.isnan(CIMS_SO2):
                Time_correct.append(t_LIF)
                LIF_mr_time_correct.append(LIF_SO2)
                FAAM_mr_time_correct.append(FAAM_SO2)
                CIMS_mr_time_correct.append(CIMS_SO2)

# look at time series 1 s
plt.plot(Time_correct, CIMS_mr_time_correct, color='green', alpha=0.7)
plt.plot(Time_correct, FAAM_mr_time_correct, color='orange', alpha=0.7)
plt.plot(Time_correct, LIF_mr_time_correct, color='blue', alpha=0.6)
plt.axhline(0, linestyle='--', label='_nolegend_')
plt.xlabel("Date time", fontsize=20)
plt.ylabel("SO$_2$ mixing ratio / ppb", fontsize=20)
plt.tick_params(axis='both', labelsize=18)

LIF_SO2_mr_10s_tm = pd.DataFrame(LIF_mr_time_correct, Time_correct)
LIF_SO2_mr_10s_tm = LIF_SO2_mr_10s_tm.resample('10s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
LIF_SO2_mr_10s_tm = LIF_SO2_mr_10s_tm.rename({0:'LIF_tm_final'}, axis=1)
LIF_SO2_mr_10s_tm.index = LIF_SO2_mr_10s_tm.index.tz_localize(None)
#LIF_SO2_mr_10s_tm.to_excel('Final_LIF_SO2_mr_10s_tm_2.xlsx')

FAAM_SO2_mr_10s_tm = pd.DataFrame(FAAM_mr_time_correct, Time_correct)
FAAM_SO2_mr_10s_tm = FAAM_SO2_mr_10s_tm.resample('10s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
FAAM_SO2_mr_10s_tm = FAAM_SO2_mr_10s_tm.rename({0:'FAAM_tm_final'}, axis=1)
FAAM_SO2_mr_10s_tm.index = FAAM_SO2_mr_10s_tm.index.rename('Time_tm_final')
FAAM_SO2_mr_10s_tm.index = FAAM_SO2_mr_10s_tm.index.tz_localize(None)
#FAAM_SO2_mr_10s_tm.to_excel('Final_FAAM_SO2_mr_10s_tm.xlsx')

CIMS_SO2_mr_10s_tm = pd.DataFrame(CIMS_mr_time_correct, Time_correct)
CIMS_SO2_mr_10s_tm = CIMS_SO2_mr_10s_tm.resample('10s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
CIMS_SO2_mr_10s_tm = CIMS_SO2_mr_10s_tm.rename({0:'CIMS_tm_final'}, axis=1)
CIMS_SO2_mr_10s_tm.index = CIMS_SO2_mr_10s_tm.index.tz_localize(None)
#CIMS_SO2_mr_10s_tm.to_excel('Final_CIMS_cps_10s_tm.xlsx')

plt.plot(LIF_SO2_mr_10s_tm, color='blue', alpha=0.6)
plt.plot(FAAM_SO2_mr_10s_tm, color='orange', alpha=0.7)
plt.plot(CIMS_SO2_mr_10s_tm/10, color='green', alpha=0.7)


## rewritten - LIF/FAAM (Fig. 8A)
plt.plot(FAAM_SO2_mr_1hz, color='orange', alpha=0.7)
plt.plot(SO2_mr_1hz, color='blue', alpha=0.6)

FAAM_mr_time_correct = []
LIF_mr_time_correct = []
Time_correct = []

for t_FAAM, t_LIF, FAAM_SO2, LIF_SO2 in zip(FAAM_SO2_mr_1hz.index[4:], SO2_mr_1hz.index[:-3], FAAM_SO2_mr_1hz['SO2_ppb'][4:], SO2_mr_1hz['SO2_ppb'][:-3]):
    if not math.isnan(LIF_SO2):
        if not math.isnan(FAAM_SO2):
            Time_correct.append(t_LIF)
            LIF_mr_time_correct.append(LIF_SO2)
            FAAM_mr_time_correct.append(FAAM_SO2)

# look at time series 1 s
plt.plot(Time_correct, FAAM_mr_time_correct, color='orange', alpha=0.7)
plt.plot(Time_correct, LIF_mr_time_correct, color='blue', alpha=0.6)
plt.axhline(0, linestyle='--', label='_nolegend_')
plt.xlabel("Date time", fontsize=20)
plt.ylabel("SO$_2$ mixing ratio / ppb", fontsize=20)
plt.tick_params(axis='both', labelsize=18)

LIF_SO2_mr_10s_tm = pd.DataFrame(LIF_mr_time_correct, Time_correct)
LIF_SO2_mr_10s_tm = LIF_SO2_mr_10s_tm.resample('10s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
LIF_SO2_mr_10s_tm = LIF_SO2_mr_10s_tm.rename({0:'LIF_tm_final'}, axis=1)
LIF_SO2_mr_10s_tm.index = LIF_SO2_mr_10s_tm.index.rename('Time_tm_final')
LIF_SO2_mr_10s_tm.index = LIF_SO2_mr_10s_tm.index.tz_localize(None)
#LIF_SO2_mr_10s_tm.to_excel('Final_LIF_FAAM_SO2_mr_10s_tm.xlsx')

FAAM_SO2_mr_10s_tm = pd.DataFrame(FAAM_mr_time_correct, Time_correct)
FAAM_SO2_mr_10s_tm = FAAM_SO2_mr_10s_tm.resample('10s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
FAAM_SO2_mr_10s_tm = FAAM_SO2_mr_10s_tm.rename({0:'FAAM_tm_final'}, axis=1)
FAAM_SO2_mr_10s_tm.index = FAAM_SO2_mr_10s_tm.index.tz_localize(None)
#FAAM_SO2_mr_10s_tm.to_excel('Final_FAAM_LIF_SO2_mr_10s_tm.xlsx')


## rewritten - LIF/CIMS (Fig. 8B)
plt.plot(CIMS_SO2_mr_1hz, color='green', alpha=0.7)
plt.plot(SO2_mr_1hz, color='blue', alpha=0.6)

# time match data and remove nans
LIF_mr_time_correct = []
CIMS_mr_time_correct = []
Time_correct = []

for t_LIF, t_CIMS, LIF_SO2, CIMS_SO2 in zip(SO2_mr_1hz.index[:-3], CIMS_SO2_mr_1hz.index[:-1], SO2_mr_1hz['SO2_ppb'][:-3], CIMS_SO2_mr_1hz['SO2_ppb'][:-1]):
    if not math.isnan(LIF_SO2):
        if not math.isnan(CIMS_SO2):
            Time_correct.append(t_LIF)
            LIF_mr_time_correct.append(LIF_SO2)
            CIMS_mr_time_correct.append(CIMS_SO2)

# look at time series 1 s
plt.plot(Time_correct, CIMS_mr_time_correct, color='green', alpha=0.7)
plt.plot(Time_correct, LIF_mr_time_correct, color='blue', alpha=0.6)
plt.axhline(0, linestyle='--', label='_nolegend_')
plt.xlabel("Date time", fontsize=20)
plt.ylabel("SO$_2$ mixing ratio / ppb", fontsize=20)
plt.tick_params(axis='both', labelsize=18)

LIF_SO2_mr_10s_tm = pd.DataFrame(LIF_mr_time_correct, Time_correct)
LIF_SO2_mr_10s_tm = LIF_SO2_mr_10s_tm.resample('10s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
LIF_SO2_mr_10s_tm = LIF_SO2_mr_10s_tm.rename({0:'LIF_tm_final'}, axis=1)
LIF_SO2_mr_10s_tm.index = LIF_SO2_mr_10s_tm.index.rename('Time_tm_final')
LIF_SO2_mr_10s_tm.index = LIF_SO2_mr_10s_tm.index.tz_localize(None)
#LIF_SO2_mr_10s_tm.to_excel('Final_LIF_CIMS_SO2_mr_10s_tm.xlsx')

CIMS_SO2_mr_10s_tm = pd.DataFrame(CIMS_mr_time_correct, Time_correct)
CIMS_SO2_mr_10s_tm = CIMS_SO2_mr_10s_tm.resample('10s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
CIMS_SO2_mr_10s_tm = CIMS_SO2_mr_10s_tm.rename({0:'CIMS_tm_final'}, axis=1)
CIMS_SO2_mr_10s_tm.index = CIMS_SO2_mr_10s_tm.index.tz_localize(None)
#CIMS_SO2_mr_10s_tm.to_excel('Final_CIMS_LIF_cps_10s_tm.xlsx')

plt.plot(LIF_SO2_mr_10s_tm, color='blue', alpha=0.6)
plt.plot(CIMS_SO2_mr_10s_tm/10, color='green', alpha=0.7)

