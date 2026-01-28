# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 08:48:16 2024

@author: lgt505
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from netCDF4 import Dataset, num2date

# Read in LIF SO2 data
data_c289 = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\C289_030522\York_LIF_SO2_10s_C289.csv', header=5)
data_c289['Date_time'] = pd.to_datetime(data_c289['Date_time'])
data_c289 = data_c289.set_index(['Date_time'], drop=True)

data_c290 = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\C290_050522\York_LIF_SO2_10s_C290.csv', header=5)
data_c290['Date_time'] = pd.to_datetime(data_c290['Date_time'])
data_c290 = data_c290.set_index(['Date_time'], drop=True)

data_c292 = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\C292_070522\York_LIF_SO2_10s_C292.csv', header=5)
data_c292['Date_time'] = pd.to_datetime(data_c292['Date_time'])
data_c292 = data_c292.set_index(['Date_time'], drop=True)

data_c293 = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\C293_080522\York_LIF_SO2_10s_C293.csv', header=5)
data_c293['Date_time'] = pd.to_datetime(data_c293['Date_time'])
data_c293 = data_c293.set_index(['Date_time'], drop=True)

data_all = pd.concat([data_c289, data_c290, data_c292, data_c293])
#data_all = data_all.resample('60s').aggregate(lambda x: pd.DataFrame.mean(x, skipna=False))
data_all = data_all['SO2_ppt']  / 1000

plt.plot(data_all)

#data_all = data_all.dropna()
data_all_median = np.median(data_all)
data_all_mean = np.mean(data_all)


# Read in altitude data
FAAM_data = Dataset(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\FAAM_data\core_faam_20220503_v005_r1_c289_1hz.nc', format='NETCDF4_CLASSIC')
nctime = FAAM_data.variables['Time'][:]
t_unit = FAAM_data.variables['Time'].units
t_cal = FAAM_data.variables['Time'].calendar
tvalue = num2date(nctime, units=t_unit, calendar=t_cal)
str_time = [i.strftime("%Y-%m-%d %H:%M:%S") for i in tvalue]

ncalt = FAAM_data.variables['ALT_GIN'][:]
Alt_df = pd.DataFrame(ncalt, str_time, columns=['Alt'])
Alt_df.index = pd.to_datetime(Alt_df.index, utc=True)
Alt_df.index = Alt_df.index.rename('Date_time')
Alt_df_10s = Alt_df.resample('10s').mean()


FAAM_data2 = Dataset(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\FAAM_data\core_faam_20220505_v005_r2_c290_1hz.nc', format='NETCDF4_CLASSIC')
nctime2 = FAAM_data2.variables['Time'][:]
t_unit2 = FAAM_data2.variables['Time'].units
t_cal2 = FAAM_data2.variables['Time'].calendar
tvalue2 = num2date(nctime2, units=t_unit2, calendar=t_cal2)
str_time2 = [i.strftime("%Y-%m-%d %H:%M:%S") for i in tvalue2]

ncalt2 = FAAM_data2.variables['ALT_GIN'][:]
Alt_df2 = pd.DataFrame(ncalt2, str_time2, columns=['Alt'])
Alt_df2.index = pd.to_datetime(Alt_df2.index, utc=True)
Alt_df2.index = Alt_df2.index.rename('Date_time')
Alt_df2_10s = Alt_df2.resample('10s').mean()


FAAM_data3 = Dataset(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\FAAM_data\core_faam_20220507_v005_r2_c292_1hz.nc', format='NETCDF4_CLASSIC')
nctime3 = FAAM_data3.variables['Time'][:]
t_unit3 = FAAM_data3.variables['Time'].units
t_cal3 = FAAM_data3.variables['Time'].calendar
tvalue3 = num2date(nctime3, units=t_unit3, calendar=t_cal3)
str_time3 = [i.strftime("%Y-%m-%d %H:%M:%S") for i in tvalue3]

ncalt3 = FAAM_data3.variables['ALT_GIN'][:]
Alt_df3 = pd.DataFrame(ncalt3, str_time3, columns=['Alt'])
Alt_df3.index = pd.to_datetime(Alt_df3.index, utc=True)
Alt_df3.index = Alt_df3.index.rename('Date_time')
Alt_df3_10s = Alt_df3.resample('10s').mean()


FAAM_data4 = Dataset(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACSIS\FAAM_data\core_faam_20220508_v005_r2_c293_1hz.nc', format='NETCDF4_CLASSIC')
nctime4 = FAAM_data4.variables['Time'][:]
t_unit4 = FAAM_data4.variables['Time'].units
t_cal4 = FAAM_data4.variables['Time'].calendar
tvalue4 = num2date(nctime4, units=t_unit4, calendar=t_cal4)
str_time4 = [i.strftime("%Y-%m-%d %H:%M:%S") for i in tvalue4]

ncalt4 = FAAM_data4.variables['ALT_GIN'][:]
Alt_df4 = pd.DataFrame(ncalt4, str_time4, columns=['Alt'])
Alt_df4.index = pd.to_datetime(Alt_df4.index, utc=True)
Alt_df4.index = Alt_df4.index.rename('Date_time')
Alt_df4_10s = Alt_df4.resample('10s').mean()

Alt_all = pd.concat([Alt_df_10s, Alt_df2_10s, Alt_df3_10s, Alt_df4_10s])
Alt_all = Alt_all['Alt']
plt.plot(Alt_all)

Alt_all.index = Alt_all.index.tz_localize(None)
combined_df2 = pd.DataFrame({'Alt_m': Alt_all, 'SO2_ppt': data_all})


nan_rows = combined_df2.isna().any(axis=1)
combined_df2[nan_rows] = np.nan
combined_df2 = combined_df2.dropna()

combined_df2['Alt_m'] = combined_df2['Alt_m'].astype(int)
altitude_bins = range(0, combined_df2['Alt_m'].max() + 100, 100)
combined_df2['Alt_bin'] = pd.cut(combined_df2['Alt_m'], bins=altitude_bins, right=False)
SO2_by_alt = combined_df2.groupby('Alt_bin')['SO2_ppt'].agg(
    mean='mean',
    median='median',
    std_error=lambda x: x.std() / np.sqrt(len(x)))     
SO2_by_alt['Alt_mid'] = [interval.mid for interval in SO2_by_alt.index]            

plt.errorbar(SO2_by_alt['mean'], SO2_by_alt['Alt_mid'], xerr=2 * SO2_by_alt['std_error'], color='steelblue')
plt.errorbar(SO2_by_alt['median'], SO2_by_alt['Alt_mid'], xerr=2 * SO2_by_alt['std_error'], color='orange')
plt.ylabel('Altitude (m)', fontsize=20)
plt.xlabel('LIF SO$_2$ mixing ratio (pptv)', fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.legend(['mean', 'median'], fontsize=16)

SO2_by_alt.to_excel('ACSIS_SO2_mr_alt_mean_median.xlsx')

