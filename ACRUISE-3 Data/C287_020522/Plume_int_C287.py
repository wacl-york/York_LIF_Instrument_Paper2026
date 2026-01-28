# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:03:19 2024

@author: lgt505
"""
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from acruisepy import peakid


CO2_int = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\C287_CO2_tm_int_4hz.csv')
CO2_int.index = pd.to_datetime(CO2_int['Date_time'])

SO2_int_LIF = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\C287_SO2_LIF_tm_int_4hz.csv')
SO2_int_LIF.index = pd.to_datetime(SO2_int_LIF['Date_time'])

#SO2_int_CIMS = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\C287_SO2_CIMS_tm_int_4hz.csv')
SO2_int_CIMS = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\C287_SO2_CIMS_tm_int_4hz_cps.csv')
SO2_int_CIMS.index = pd.to_datetime(SO2_int_CIMS['Date_time'])

SO2_int_PF = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\C287_SO2_PF_tm_int_1s.csv')
SO2_int_PF.index = pd.to_datetime(SO2_int_PF['Date_time'])

fig, ax = plt.subplots()
ax.plot(SO2_int_LIF['SO2_ppb'], color='blue')
ax.plot(SO2_int_CIMS['SO2_ppb'], color='green')
ax.plot(SO2_int_PF['SO2_ppb'], color='orange')
ax2 = ax.twinx()
ax2.plot(CO2_int['CO2_ppm'], color='red')


###################### Picarro CO2
CO2_bkd = pd.DataFrame(CO2_int['CO2_ppm'])

CO2_bkd.loc['2022-05-02 15:17:57':'2022-05-02 15:18:55'] = np.nan
CO2_bkd = CO2_bkd.interpolate(method='linear')
plt.plot(CO2_bkd)

#CO2_remove = pd.DataFrame(CO2_int) - CO2_bkd
#plt.plot(CO2_remove)

CO2_bkd['ema'] = CO2_bkd['CO2_ppm'].ewm(alpha=0.01).mean()
CO2_bkd['sma'] = CO2_bkd['CO2_ppm'].rolling(100, center=False).mean()
CO2_bkd.plot()

CO2_remove = pd.DataFrame({'CO2_ppm': CO2_int['CO2_ppm'] - CO2_bkd['ema']})
plt.plot(CO2_remove)

CO2_remove_loc = CO2_remove['CO2_ppm'].loc['2022-05-02 15:17:57':'2022-05-02 15:18:55']
areas_CO2 = np.trapz(CO2_remove_loc, dx=0.25)


#### error
change_in_x = 0.25
CO2_remove_err = (CO2_remove_loc * 0) + (2 * 0.574)**2
CO2_remove_err[0] = CO2_remove_err[0] * 0.25
CO2_remove_err[-1] = CO2_remove_err[-1] * 0.25
overall_CO2_err = np.sqrt(np.sum(CO2_remove_err * (change_in_x**2)))


###################### SO2 LIF
SO2_bkd_LIF = pd.DataFrame(SO2_int_LIF['SO2_ppb'])

SO2_bkd_LIF.loc['2022-05-02 15:17:57':'2022-05-02 15:18:57'] = np.nan
SO2_bkd_LIF = SO2_bkd_LIF.interpolate(method='linear')
plt.plot(SO2_bkd_LIF)

#SO2_remove_LIF = pd.DataFrame(SO2_int_LIF) - SO2_bkd_LIF
#plt.plot(SO2_remove_LIF)

SO2_bkd_LIF['ema'] = SO2_bkd_LIF['SO2_ppb'].ewm(alpha=0.01).mean()
SO2_bkd_LIF['sma'] = SO2_bkd_LIF['SO2_ppb'].rolling(100, center=False).mean()
SO2_bkd_LIF.plot()

SO2_remove_LIF = pd.DataFrame({'SO2_ppb': SO2_int_LIF['SO2_ppb'] - SO2_bkd_LIF['ema']})
plt.plot(SO2_remove_LIF)

SO2_remove_LIF_loc = SO2_remove_LIF['SO2_ppb'].loc['2022-05-02 15:17:57':'2022-05-02 15:18:57']
areas_SO2_LIF = np.trapz(SO2_remove_LIF_loc, dx=0.25)


#### error
LIF_err = 0.1
change_in_x = 0.25
SO2_remove_LIF_err = (SO2_remove_LIF_loc * LIF_err)**2
SO2_remove_LIF_err[0] = SO2_remove_LIF_err[0] * 0.25
SO2_remove_LIF_err[-1] = SO2_remove_LIF_err[-1] * 0.25
overall_LIF_err = np.sqrt(np.sum(SO2_remove_LIF_err * (change_in_x**2)))


###################### SO2 CIMS
SO2_bkd_CIMS = pd.DataFrame(SO2_int_CIMS['SO2_ppb'])

SO2_bkd_CIMS.loc['2022-05-02 15:17:55':'2022-05-02 15:18:57'] = np.nan
SO2_bkd_CIMS = SO2_bkd_CIMS.interpolate(method='linear')
plt.plot(SO2_bkd_CIMS, color='orange')

#SO2_remove_CIMS = pd.DataFrame(SO2_int_CIMS) - SO2_bkd_CIMS
#plt.plot(SO2_remove_CIMS)

SO2_bkd_CIMS['ema'] = SO2_bkd_CIMS['SO2_ppb'].ewm(alpha=0.01).mean()
SO2_bkd_CIMS['sma'] = SO2_bkd_CIMS['SO2_ppb'].rolling(100, center=False).mean()
SO2_bkd_CIMS.plot()

SO2_remove_CIMS = pd.DataFrame({'SO2_ppb': SO2_int_CIMS['SO2_ppb'] - SO2_bkd_CIMS['ema']})
plt.plot(SO2_remove_CIMS)

SO2_remove_CIMS_loc = SO2_remove_CIMS['SO2_ppb'].loc['2022-05-02 15:17:55':'2022-05-02 15:18:57']
areas_SO2_CIMS = np.trapz(SO2_remove_CIMS_loc, dx=0.25)


#### error
CIMS_err = 1.02
change_in_x = 0.25
SO2_remove_CIMS_err = (SO2_remove_CIMS_loc * CIMS_err)**2
SO2_remove_CIMS_err[0] = SO2_remove_CIMS_err[0] * 0.25
SO2_remove_CIMS_err[-1] = SO2_remove_CIMS_err[-1] * 0.25
overall_CIMS_err = np.sqrt(np.sum(SO2_remove_CIMS_err * (change_in_x**2)))


###################### SO2 PF
SO2_bkd_PF = pd.DataFrame(SO2_int_PF['SO2_ppb'])

SO2_bkd_PF.loc['2022-05-02 15:17:56':'2022-05-02 15:19:10'] = np.nan
SO2_bkd_PF = SO2_bkd_PF.interpolate(method='linear')
plt.plot(SO2_bkd_PF)

#SO2_remove_PF = pd.DataFrame(SO2_int_PF) - SO2_bkd_PF
#plt.plot(SO2_remove_PF)

SO2_bkd_PF['ema'] = SO2_bkd_PF['SO2_ppb'].ewm(alpha=0.01).mean()
SO2_bkd_PF['sma'] = SO2_bkd_PF['SO2_ppb'].rolling(100, center=False).mean()
SO2_bkd_PF.plot()

SO2_remove_PF = pd.DataFrame({'SO2_ppb': SO2_int_PF['SO2_ppb'] - SO2_bkd_PF['ema']})
plt.plot(SO2_remove_PF)

SO2_remove_PF_loc = SO2_remove_PF['SO2_ppb'].loc['2022-05-02 15:17:56':'2022-05-02 15:19:10']
areas_SO2_PF = np.trapz(SO2_remove_PF_loc, dx=1)


#### error
PF_err = 0.18
change_in_x = 1
SO2_remove_PF_err = (SO2_remove_PF_loc * PF_err)**2
SO2_remove_PF_err[0] = SO2_remove_PF_err[0] * 0.25
SO2_remove_PF_err[-1] = SO2_remove_PF_err[-1] * 0.25
overall_PF_err = np.sqrt(np.sum(SO2_remove_PF_err * (change_in_x**2)))


#### Integration emission ratios
# SO2 LIF:CO2
LIF_CO2 = areas_SO2_LIF / areas_CO2
LIF_CO2_err = (areas_SO2_LIF / areas_CO2) * np.sqrt((overall_LIF_err / areas_SO2_LIF)**2 + (overall_CO2_err / areas_CO2)**2)

# SO2 PF:CO2
PF_CO2 = areas_SO2_PF / areas_CO2
PF_CO2_err = (areas_SO2_PF / areas_CO2) * np.sqrt((overall_PF_err / areas_SO2_PF)**2 + (overall_CO2_err / areas_CO2)**2)

# SO2 CIMS:CO2
CIMS_CO2 = areas_SO2_CIMS / areas_CO2
CIMS_CO2_err = (areas_SO2_CIMS / areas_CO2) * np.sqrt((overall_CIMS_err / areas_SO2_CIMS)**2 + (overall_CO2_err / areas_CO2)**2)




################### Regression
#SO2_remove_CIMS = (areas_SO2_LIF / areas_SO2_CIMS) * SO2_remove_CIMS.loc['2022-05-02 15:17:57':'2022-05-02 15:18:55']
SO2_remove_CIMS = SO2_remove_CIMS.loc['2022-05-02 15:17:57':'2022-05-02 15:18:55']
CO2_remove = CO2_remove.loc['2022-05-02 15:17:57':'2022-05-02 15:18:55']
SO2_remove_LIF = SO2_remove_LIF.loc['2022-05-02 15:17:57':'2022-05-02 15:18:55']

SO2_remove_CIMS.index = SO2_remove_CIMS.index.tz_localize(None)
CO2_remove.index = CO2_remove.index.tz_localize(None)
SO2_remove_LIF.index = SO2_remove_LIF.index.tz_localize(None)

SO2_remove_CIMS.to_excel('SO2_remove_CIMS.xlsx')
CO2_remove.to_excel('CO2_remove.xlsx')
SO2_remove_LIF.to_excel('SO2_remove_LIF.xlsx')

fig, ax = plt.subplots()
ax.plot(SO2_remove_LIF['SO2_ppb'], color='blue')
ax.plot(SO2_remove_CIMS['SO2_ppb'], color='green')
ax2 = ax.twinx()
ax2.plot(CO2_remove['CO2_ppm'], color='orange')


## CIMS
plt.scatter(CO2_remove['CO2_ppm'], SO2_remove_CIMS['SO2_ppb'])
scipy.stats.linregress(CO2_remove['CO2_ppm'], SO2_remove_CIMS['SO2_ppb'])


## LIF
plt.scatter(CO2_remove['CO2_ppm'], SO2_remove_LIF['SO2_ppb'])
scipy.stats.linregress(CO2_remove['CO2_ppm'], SO2_remove_LIF['SO2_ppb'])





############################# plume 2 #########################################

CO2_int = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\C287_CO2_tm_int_4hz_2.csv')
CO2_int.index = pd.to_datetime(CO2_int['Date_time'])

SO2_int_LIF = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\C287_SO2_LIF_tm_int_4hz_2.csv')
SO2_int_LIF.index = pd.to_datetime(SO2_int_LIF['Date_time'])

#SO2_int_CIMS = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\C287_SO2_CIMS_tm_int_4hz_2.csv')
SO2_int_CIMS = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\C287_SO2_CIMS_tm_int_4hz_2_cps.csv')
SO2_int_CIMS.index = pd.to_datetime(SO2_int_CIMS['Date_time'])

SO2_int_PF = pd.read_csv(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\C287_SO2_PF_tm_int_1s_2.csv')
SO2_int_PF.index = pd.to_datetime(SO2_int_PF['Date_time'])

fig, ax = plt.subplots()
ax.plot(SO2_int_LIF['SO2_ppb'], color='blue')
ax.plot(SO2_int_CIMS['SO2_ppb'], color='green')
ax.plot(SO2_int_PF['SO2_ppb'], color='orange')
ax2 = ax.twinx()
ax2.plot(CO2_int['CO2_ppm'], color='red')


###################### Picarro CO2
CO2_bkd = pd.DataFrame(CO2_int['CO2_ppm'])

CO2_bkd.loc['2022-05-02 15:21:05':'2022-05-02 15:23:40'] = np.nan
CO2_bkd = CO2_bkd.interpolate(method='linear')
plt.plot(CO2_bkd)

#CO2_remove = pd.DataFrame(CO2_int) - CO2_bkd
#plt.plot(CO2_remove)

CO2_bkd['ema'] = CO2_bkd['CO2_ppm'].ewm(alpha=0.01).mean()
CO2_bkd['sma'] = CO2_bkd['CO2_ppm'].rolling(100, center=False).mean()
CO2_bkd.plot()

CO2_remove = pd.DataFrame({'CO2_ppm': CO2_int['CO2_ppm'] - CO2_bkd['ema']})
plt.plot(CO2_remove)

CO2_remove_loc = CO2_remove['CO2_ppm'].loc['2022-05-02 15:21:05':'2022-05-02 15:23:40']
areas_CO2 = np.trapz(CO2_remove_loc, dx=0.25)


#### error
change_in_x = 0.25
CO2_remove_err = (CO2_remove_loc * 0) + (2 * 0.574)**2
CO2_remove_err[0] = CO2_remove_err[0] * 0.25
CO2_remove_err[-1] = CO2_remove_err[-1] * 0.25
overall_CO2_err = np.sqrt(np.sum(CO2_remove_err * (change_in_x**2)))



###################### SO2 LIF
SO2_bkd_LIF = pd.DataFrame(SO2_int_LIF['SO2_ppb'])

SO2_bkd_LIF.loc['2022-05-02 15:21:05':'2022-05-02 15:23:43'] = np.nan
SO2_bkd_LIF = SO2_bkd_LIF.interpolate(method='linear')
plt.plot(SO2_bkd_LIF)

#SO2_remove_LIF = pd.DataFrame(SO2_int_LIF) - SO2_bkd_LIF
#plt.plot(SO2_remove_LIF)

SO2_bkd_LIF['ema'] = SO2_bkd_LIF['SO2_ppb'].ewm(alpha=0.01).mean()
SO2_bkd_LIF['sma'] = SO2_bkd_LIF['SO2_ppb'].rolling(100, center=False).mean()
SO2_bkd_LIF.plot()

SO2_remove_LIF = pd.DataFrame({'SO2_ppb': SO2_int_LIF['SO2_ppb'] - SO2_bkd_LIF['ema']})
plt.plot(SO2_remove_LIF)

SO2_remove_LIF_loc = SO2_remove_LIF['SO2_ppb'].loc['2022-05-02 15:21:05':'2022-05-02 15:23:43']
areas_SO2_LIF = np.trapz(SO2_remove_LIF_loc, dx=0.25)


#### error
LIF_err = 0.1
change_in_x = 0.25
SO2_remove_LIF_err = (SO2_remove_LIF_loc * LIF_err)**2
SO2_remove_LIF_err[0] = SO2_remove_LIF_err[0] * 0.25
SO2_remove_LIF_err[-1] = SO2_remove_LIF_err[-1] * 0.25
overall_LIF_err = np.sqrt(np.sum(SO2_remove_LIF_err * (change_in_x**2)))



###################### SO2 CIMS
SO2_bkd_CIMS = pd.DataFrame(SO2_int_CIMS['SO2_ppb'])

SO2_bkd_CIMS.loc['2022-05-02 15:21:05':'2022-05-02 15:23:42'] = np.nan
SO2_bkd_CIMS = SO2_bkd_CIMS.interpolate(method='linear')
plt.plot(SO2_bkd_CIMS)

#SO2_remove_CIMS = pd.DataFrame(SO2_int_CIMS) - SO2_bkd_CIMS
#plt.plot(SO2_remove_CIMS)

SO2_bkd_CIMS['ema'] = SO2_bkd_CIMS['SO2_ppb'].ewm(alpha=0.01).mean()
SO2_bkd_CIMS['sma'] = SO2_bkd_CIMS['SO2_ppb'].rolling(100, center=False).mean()
SO2_bkd_CIMS.plot()

SO2_remove_CIMS = pd.DataFrame({'SO2_ppb': SO2_int_CIMS['SO2_ppb'] - SO2_bkd_CIMS['ema']})
plt.plot(SO2_remove_CIMS)

SO2_remove_CIMS_loc = SO2_remove_CIMS['SO2_ppb'].loc['2022-05-02 15:21:05':'2022-05-02 15:23:42']
areas_SO2_CIMS = np.trapz(SO2_remove_CIMS_loc, dx=0.25)


#### error
CIMS_err = 1.02
change_in_x = 0.25
SO2_remove_CIMS_err = (SO2_remove_CIMS_loc * CIMS_err)**2
SO2_remove_CIMS_err[0] = SO2_remove_CIMS_err[0] * 0.25
SO2_remove_CIMS_err[-1] = SO2_remove_CIMS_err[-1] * 0.25
overall_CIMS_err = np.sqrt(np.sum(SO2_remove_CIMS_err * (change_in_x**2)))



###################### SO2 PF
SO2_bkd_PF = pd.DataFrame(SO2_int_PF['SO2_ppb'])

SO2_bkd_PF.loc['2022-05-02 15:21:00':'2022-05-02 15:23:57'] = np.nan
SO2_bkd_PF = SO2_bkd_PF.interpolate(method='linear')
plt.plot(SO2_bkd_PF)

#SO2_remove_PF = pd.DataFrame(SO2_int_PF) - SO2_bkd_PF
#plt.plot(SO2_remove_PF)

SO2_bkd_PF['ema'] = SO2_bkd_PF['SO2_ppb'].ewm(alpha=0.01).mean()
SO2_bkd_PF['sma'] = SO2_bkd_PF['SO2_ppb'].rolling(100, center=False).mean()
SO2_bkd_PF.plot()

SO2_remove_PF = pd.DataFrame({'SO2_ppb': SO2_int_PF['SO2_ppb'] - SO2_bkd_PF['ema']})
plt.plot(SO2_remove_PF)

SO2_remove_PF_loc = SO2_remove_PF['SO2_ppb'].loc['2022-05-02 15:21:00':'2022-05-02 15:23:57']
areas_SO2_PF = np.trapz(SO2_remove_PF_loc, dx=1)


#### error
PF_err = 0.18
change_in_x = 1
SO2_remove_PF_err = (SO2_remove_PF_loc * PF_err)**2
SO2_remove_PF_err[0] = SO2_remove_PF_err[0] * 0.25
SO2_remove_PF_err[-1] = SO2_remove_PF_err[-1] * 0.25
overall_PF_err = np.sqrt(np.sum(SO2_remove_PF_err * (change_in_x**2)))



#### Integration emission ratios
# SO2 LIF:CO2
LIF_CO2 = areas_SO2_LIF / areas_CO2
LIF_CO2_err = (areas_SO2_LIF / areas_CO2) * np.sqrt((overall_LIF_err / areas_SO2_LIF)**2 + (overall_CO2_err / areas_CO2)**2)

# SO2 PF:CO2
PF_CO2 = areas_SO2_PF / areas_CO2
PF_CO2_err = (areas_SO2_PF / areas_CO2) * np.sqrt((overall_PF_err / areas_SO2_PF)**2 + (overall_CO2_err / areas_CO2)**2)

# SO2 CIMS:CO2
CIMS_CO2 = areas_SO2_CIMS / areas_CO2
CIMS_CO2_err = (areas_SO2_CIMS / areas_CO2) * np.sqrt((overall_CIMS_err / areas_SO2_CIMS)**2 + (overall_CO2_err / areas_CO2)**2)




################### Regression
#SO2_remove_CIMS = (areas_SO2_LIF / areas_SO2_CIMS) * SO2_remove_CIMS.loc['2022-05-02 15:21:05':'2022-05-02 15:23:40']
SO2_remove_CIMS = SO2_remove_CIMS.loc['2022-05-02 15:21:05':'2022-05-02 15:23:40']
CO2_remove = CO2_remove.loc['2022-05-02 15:21:05':'2022-05-02 15:23:40']
SO2_remove_LIF = SO2_remove_LIF.loc['2022-05-02 15:21:05':'2022-05-02 15:23:40']

SO2_remove_CIMS.index = SO2_remove_CIMS.index.tz_localize(None)
CO2_remove.index = CO2_remove.index.tz_localize(None)
SO2_remove_LIF.index = SO2_remove_LIF.index.tz_localize(None)

SO2_remove_CIMS.to_excel('SO2_remove_CIMS_2.xlsx')
CO2_remove.to_excel('CO2_remove_2.xlsx')
SO2_remove_LIF.to_excel('SO2_remove_LIF_2.xlsx')

fig, ax = plt.subplots()
ax.plot(SO2_remove_LIF['SO2_ppb'], color='blue')
ax.plot(SO2_remove_CIMS['SO2_ppb'], color='green')
ax2 = ax.twinx()
ax2.plot(CO2_remove['CO2_ppm'], color='orange')


## CIMS
plt.scatter(CO2_remove['CO2_ppm'], SO2_remove_CIMS['SO2_ppb'])
scipy.stats.linregress(CO2_remove['CO2_ppm'], SO2_remove_CIMS['SO2_ppb'])


## LIF
plt.scatter(CO2_remove['CO2_ppm'], SO2_remove_LIF['SO2_ppb'])
scipy.stats.linregress(CO2_remove['CO2_ppm'], SO2_remove_LIF['SO2_ppb'])


