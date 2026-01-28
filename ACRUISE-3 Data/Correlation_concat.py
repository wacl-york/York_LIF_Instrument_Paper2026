# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:52:46 2023

@author: lgt505
"""
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from datetime import datetime as dt
import glob
import matplotlib.pyplot as plt
import scipy.optimize

#LIF_C284 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE_ACSIS\C284_290422\LIF_time_avg_all.xlsx')
#FAAM_C284 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE_ACSIS\C284_290422\FAAM_time_avg_all.xlsx')

LIF_C285 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\Data_paper\Final_LIF_SO2_mr_10s_tm_2.xlsx')
FAAM_C285 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\Data_paper\Final_FAAM_SO2_mr_10s_tm.xlsx')
CIMS_C285 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\Data_paper\Final_CIMS_SO2_mr_10s_tm.xlsx')
CIMS_C285_uncorr = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\Data_paper\Final_CIMS_SO2_mr_10s_tm_uncorrected.xlsx')
CIMS_C285_cps = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\Data_paper\Final_CIMS_cps_10s_tm.xlsx')

LIF_C286 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C286_010522\Data_paper\Final_LIF_SO2_mr_10s_tm_2.xlsx')
FAAM_C286 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C286_010522\Data_paper\Final_FAAM_SO2_mr_10s_tm.xlsx')
CIMS_C286 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C286_010522\Data_paper\Final_CIMS_SO2_mr_10s_tm.xlsx')
CIMS_C286_uncorr = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C286_010522\Data_paper\Final_CIMS_SO2_mr_10s_tm_uncorrected.xlsx')
CIMS_C286_cps = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C286_010522\Data_paper\Final_CIMS_cps_10s_tm.xlsx')

LIF_C287 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\Data_paper\Final_LIF_SO2_mr_10s_tm_2.xlsx')
FAAM_C287 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\Data_paper\Final_FAAM_SO2_mr_10s_tm.xlsx')
CIMS_C287 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\Data_paper\Final_CIMS_SO2_mr_10s_tm.xlsx')
CIMS_C287_uncorr = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\Data_paper\Final_CIMS_SO2_mr_10s_tm_uncorrected.xlsx')
CIMS_C287_cps = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\Data_paper\Final_CIMS_cps_10s_tm.xlsx')


LIF_all = [LIF_C285, LIF_C286, LIF_C287]

FAAM_all = [FAAM_C285, FAAM_C286, FAAM_C287]

CIMS_all = [CIMS_C285, CIMS_C286, CIMS_C287]

CIMS_uncorr = [CIMS_C285_uncorr, CIMS_C286_uncorr, CIMS_C287_uncorr]

CIMS_cps = [CIMS_C285_cps, CIMS_C286_cps, CIMS_C287_cps]

LIF_all_concat = pd.concat(LIF_all, axis=0, ignore_index=True)
FAAM_all_concat = pd.concat(FAAM_all, axis=0, ignore_index=True)
CIMS_all_concat = pd.concat(CIMS_all, axis=0, ignore_index=True)
CIMS_uncorr_concat = pd.concat(CIMS_uncorr, axis=0, ignore_index=True)
CIMS_cps_concat = pd.concat(CIMS_cps, axis=0, ignore_index=True)

LIF_all_concat.index = pd.to_datetime(LIF_all_concat['Unnamed: 0'], format="%Y/%m/%d %H:%M:%S", utc=True)
FAAM_all_concat.index = pd.to_datetime(FAAM_all_concat['Time_tm_final'], format="%Y/%m/%d %H:%M:%S", utc=True)
CIMS_all_concat.index = pd.to_datetime(CIMS_all_concat['Unnamed: 0'], format="%Y/%m/%d %H:%M:%S", utc=True)
CIMS_cps_concat.index = pd.to_datetime(CIMS_cps_concat['Unnamed: 0'], format="%Y/%m/%d %H:%M:%S", utc=True)


LIF_all_concat = LIF_all_concat.dropna()
CIMS_all_concat = CIMS_all_concat.dropna()
FAAM_all_concat = FAAM_all_concat.dropna()
CIMS_cps_concat = CIMS_cps_concat.dropna()

count_below_threshold = (FAAM_all_concat['FAAM_tm_final'] < 0.4).sum()


plt.plot(CIMS_all_concat['CIMS_tm_final'], color='green')
plt.plot(FAAM_all_concat['FAAM_tm_final'], color='orange')
plt.plot(LIF_all_concat['LIF_tm_final'], color='blue')
plt.axhline(2, color='green', linestyle='dashed')
plt.axhline(0.4, color='orange', linestyle='dashed')
plt.axhline(0.06, color='blue', linestyle='dashed')

#LIF_all_concat.to_excel('LIF_all_concat.xlsx')
#FAAM_all_concat.to_excel('FAAM_all_concat.xlsx')
#CIMS_all_concat.to_excel('CIMS_all_concat.xlsx')
#LIF_CIMS_concat.to_excel('LIF_CIMS_concat.xlsx')


# for CIMS-LIF plot
Corr_both = pd.DataFrame([LIF_all_concat['LIF_tm_final'], CIMS_all_concat['CIMS_tm_final']])
Corr_both = Corr_both.transpose()

for i in Corr_both.index:
    if Corr_both['CIMS_tm_final'][i] < 2:
        Corr_both['CIMS_tm_final'][i] = np.nan
        Corr_both['LIF_tm_final'][i] = np.nan
        
for i in Corr_both.index:
    if Corr_both['LIF_tm_final'][i] < 2:
        Corr_both['CIMS_tm_final'][i] = np.nan
        Corr_both['LIF_tm_final'][i] = np.nan
    

plt.scatter(Corr_both['LIF_tm_final'], Corr_both['CIMS_tm_final'])
Corr_both = Corr_both.dropna()
#Corr_both.to_excel('Correlation_CIMS_LIF_2ppb_updated.xlsx')

def func(x, a, b):
    return a * x + b
#scipy.optimize.curve_fit(func, Corr_both['LIF_tm_final'], Corr_both['CIMS_tm_final'])
popt, pcov = scipy.optimize.curve_fit(func, Corr_both['LIF_tm_final'], Corr_both['CIMS_tm_final'])
residuals = Corr_both['CIMS_tm_final']- func(Corr_both['LIF_tm_final'], *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Corr_both['CIMS_tm_final']-np.mean(Corr_both['CIMS_tm_final']))**2)
r_squared = 1 - (ss_res / ss_tot)


# for CIMSuncorr-LIF plot
Corr_both = pd.DataFrame([LIF_all_concat['LIF_tm_final'], CIMS_uncorr_concat['CIMS_tm_final']])
Corr_both = Corr_both.transpose()

plt.scatter(Corr_both['LIF_tm_final'], Corr_both['CIMS_tm_final'])

for i in Corr_both.index:
    if Corr_both['CIMS_tm_final'][i] < 2:
        Corr_both['CIMS_tm_final'][i] = np.nan
        Corr_both['LIF_tm_final'][i] = np.nan
        
for i in Corr_both.index:
    if Corr_both['LIF_tm_final'][i] < 2:
        Corr_both['CIMS_tm_final'][i] = np.nan
        Corr_both['LIF_tm_final'][i] = np.nan
    

plt.scatter(Corr_both['LIF_tm_final'], Corr_both['CIMS_tm_final'])
Corr_both = Corr_both.dropna()
#Corr_both.to_excel('Correlation_CIMS_uncorr_LIF_2ppb_updated.xlsx')

def func(x, a, b):
    return a * x + b
#scipy.optimize.curve_fit(func, Corr_both['LIF_tm_final'], Corr_both['CIMS_tm_final'])
popt, pcov = scipy.optimize.curve_fit(func, Corr_both['LIF_tm_final'], Corr_both['CIMS_tm_final'])
residuals = Corr_both['CIMS_tm_final']- func(Corr_both['LIF_tm_final'], *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Corr_both['CIMS_tm_final']-np.mean(Corr_both['CIMS_tm_final']))**2)
r_squared = 1 - (ss_res / ss_tot)



# for FAAM-LIF plot
Corr_both = pd.DataFrame([LIF_all_concat['LIF_tm_final'], FAAM_all_concat['FAAM_tm_final']])
Corr_both = Corr_both.transpose()

plt.scatter(Corr_both['LIF_tm_final'], Corr_both['FAAM_tm_final'])

for i in Corr_both.index:
    if Corr_both['FAAM_tm_final'][i] < 0.4:
        Corr_both['FAAM_tm_final'][i] = np.nan
        Corr_both['LIF_tm_final'][i] = np.nan
        
for i in Corr_both.index:
    if Corr_both['LIF_tm_final'][i] < 0.4:
        Corr_both['FAAM_tm_final'][i] = np.nan
        Corr_both['LIF_tm_final'][i] = np.nan

plt.scatter(Corr_both['LIF_tm_final'], Corr_both['FAAM_tm_final'])
Corr_both = Corr_both.dropna()
#Corr_both.to_excel('Correlation_FAAM_LIF_400ppt_updated.xlsx')

def func(x, a, b):
    return a * x + b
#scipy.optimize.curve_fit(func, Corr_both['LIF_tm_final'], Corr_both['CIMS_tm_final'])
popt, pcov = scipy.optimize.curve_fit(func, Corr_both['LIF_tm_final'], Corr_both['FAAM_tm_final'])
residuals = Corr_both['FAAM_tm_final']- func(Corr_both['LIF_tm_final'], *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Corr_both['FAAM_tm_final']-np.mean(Corr_both['FAAM_tm_final']))**2)
r_squared = 1 - (ss_res / ss_tot)


### rewritten

LIF_FAAM_C285 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\Data_paper\New\Final_LIF_FAAM_SO2_mr_10s_tm.xlsx')
FAAM_LIF_C285 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\Data_paper\New\Final_FAAM_LIF_SO2_mr_10s_tm.xlsx')
LIF_CIMS_C285 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\Data_paper\New\Final_LIF_CIMS_SO2_mr_10s_tm.xlsx')
CIMS_LIF_C285 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C285_300422\Data_paper\New\Final_CIMS_LIF_cps_10s_tm.xlsx')

LIF_FAAM_C286 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C286_010522\Data_paper\New\Final_LIF_FAAM_SO2_mr_10s_tm.xlsx')
FAAM_LIF_C286 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C286_010522\Data_paper\New\Final_FAAM_LIF_SO2_mr_10s_tm.xlsx')
LIF_CIMS_C286 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C286_010522\Data_paper\New\Final_LIF_CIMS_SO2_mr_10s_tm.xlsx')
CIMS_LIF_C286 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C286_010522\Data_paper\New\Final_CIMS_LIF_cps_10s_tm.xlsx')

LIF_FAAM_C287 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\Data_paper\New\Final_LIF_FAAM_SO2_mr_10s_tm.xlsx')
FAAM_LIF_C287 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\Data_paper\New\Final_FAAM_LIF_SO2_mr_10s_tm.xlsx')
LIF_CIMS_C287 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\Data_paper\New\Final_LIF_CIMS_SO2_mr_10s_tm.xlsx')
CIMS_LIF_C287 = pd.read_excel(r'C:\Users\lgt505\Google Drive\Flight Campaigns\ACRUISE\C287_020522\Data_paper\New\Final_CIMS_LIF_cps_10s_tm.xlsx')

LIF_FAAM_all = [LIF_FAAM_C285, LIF_FAAM_C286, LIF_FAAM_C287]

FAAM_LIF_all = [FAAM_LIF_C285, FAAM_LIF_C286, FAAM_LIF_C287]

LIF_CIMS_all = [LIF_CIMS_C285, LIF_CIMS_C286, LIF_CIMS_C287]

CIMS_LIF_all = [CIMS_LIF_C285, CIMS_LIF_C286, CIMS_LIF_C287]

LIF_FAAM_all_concat = pd.concat(LIF_FAAM_all, axis=0, ignore_index=True)
FAAM_LIF_all_concat = pd.concat(FAAM_LIF_all, axis=0, ignore_index=True)
LIF_CIMS_all_concat = pd.concat(LIF_CIMS_all, axis=0, ignore_index=True)
CIMS_LIF_all_concat = pd.concat(CIMS_LIF_all, axis=0, ignore_index=True)


LIF_FAAM_all_concat.index = pd.to_datetime(LIF_FAAM_all_concat['Time_tm_final'], format="%Y/%m/%d %H:%M:%S", utc=True)
FAAM_LIF_all_concat.index = pd.to_datetime(FAAM_LIF_all_concat['Unnamed: 0'], format="%Y/%m/%d %H:%M:%S", utc=True)
LIF_CIMS_all_concat.index = pd.to_datetime(LIF_CIMS_all_concat['Time_tm_final'], format="%Y/%m/%d %H:%M:%S", utc=True)
CIMS_LIF_all_concat.index = pd.to_datetime(CIMS_LIF_all_concat['Unnamed: 0'], format="%Y/%m/%d %H:%M:%S", utc=True)

LIF_FAAM_all_concat = LIF_FAAM_all_concat.dropna()
FAAM_LIF_all_concat = FAAM_LIF_all_concat.dropna()
LIF_CIMS_all_concat = LIF_CIMS_all_concat.dropna()
CIMS_LIF_all_concat = CIMS_LIF_all_concat.dropna()

count_below_threshold = (FAAM_all_concat['FAAM_tm_final'] < 0.4).sum()


plt.plot(LIF_FAAM_all_concat['LIF_tm_final'], color='blue')
plt.plot(FAAM_LIF_all_concat['FAAM_tm_final'], color='orange')
plt.plot(LIF_CIMS_all_concat['LIF_tm_final'], color='pink')
plt.plot(CIMS_LIF_all_concat['CIMS_tm_final'], color='green')
plt.axhline(2, color='green', linestyle='dashed')
plt.axhline(0.4, color='orange', linestyle='dashed')
plt.axhline(0.06, color='blue', linestyle='dashed')


### LIF-FAAM
Corr_LIF_FAAM = pd.DataFrame([LIF_FAAM_all_concat['LIF_tm_final'], FAAM_LIF_all_concat['FAAM_tm_final']])
Corr_LIF_FAAM = Corr_LIF_FAAM.transpose()

for i in Corr_LIF_FAAM.index:
    if Corr_LIF_FAAM['FAAM_tm_final'][i] < 0.4:
        Corr_LIF_FAAM['FAAM_tm_final'][i] = np.nan
        Corr_LIF_FAAM['LIF_tm_final'][i] = np.nan
        
Corr_LIF_FAAM['LIF_tm_final_err'] = Corr_LIF_FAAM['LIF_tm_final'] * (10/100) + 0.0065
Corr_LIF_FAAM['FAAM_tm_final_err'] = Corr_LIF_FAAM['FAAM_tm_final'] * (18/100)
Corr_LIF_FAAM = Corr_LIF_FAAM.dropna()

# Using York regression
from York_regression_script import York_reg
b, b_err, a, a_err, stat1, stat2 = York_reg(x=Corr_LIF_FAAM['LIF_tm_final'], y=Corr_LIF_FAAM['FAAM_tm_final'],
         sx=Corr_LIF_FAAM['LIF_tm_final_err'], sy=Corr_LIF_FAAM['FAAM_tm_final_err'],
         Ri=0, b0=1)

# Using orthogonal distance regression
from scipy.odr import ODR, Model, RealData
def f(beta, x):
    m, p = beta
    return m*x + p
data = RealData(x=Corr_LIF_FAAM['LIF_tm_final'], y=Corr_LIF_FAAM['FAAM_tm_final'],
         sx=Corr_LIF_FAAM['LIF_tm_final_err'], sy=Corr_LIF_FAAM['FAAM_tm_final_err'])
model = Model(f)
odr = ODR(data, model, beta0=[1., 0.])
output = odr.run()
output.pprint()

# Using OLS
def func(x, a, b):
    return a * x + b
#scipy.optimize.curve_fit(func, Corr_both['LIF_tm_final'], Corr_both['CIMS_tm_final'])
popt, pcov = scipy.optimize.curve_fit(func, Corr_LIF_FAAM['LIF_tm_final'], Corr_LIF_FAAM['FAAM_tm_final'])
residuals = Corr_LIF_FAAM['FAAM_tm_final']- func(Corr_LIF_FAAM['LIF_tm_final'], *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Corr_LIF_FAAM['FAAM_tm_final']-np.mean(Corr_LIF_FAAM['FAAM_tm_final']))**2)
r_squared = 1 - (ss_res / ss_tot)

# Using Theil-Sen
from scipy.stats import theilslopes

slope, intercept, lo_slope, hi_slope = theilslopes(Corr_LIF_FAAM['FAAM_tm_final'], Corr_LIF_FAAM['LIF_tm_final'],
                                                   alpha=0.95)


plt.scatter(Corr_LIF_FAAM['LIF_tm_final'], Corr_LIF_FAAM['FAAM_tm_final'])
plt.plot(Corr_LIF_FAAM['LIF_tm_final'], (Corr_LIF_FAAM['LIF_tm_final']*popt[0] + popt[1]))
plt.plot(Corr_LIF_FAAM['LIF_tm_final'], (Corr_LIF_FAAM['LIF_tm_final']*b + a), color='orange', label='York')
plt.plot(Corr_LIF_FAAM['LIF_tm_final'], (Corr_LIF_FAAM['LIF_tm_final']*output.beta[0] + output.beta[1]), color='green', label='ODR')
plt.plot(Corr_LIF_FAAM['LIF_tm_final'], (Corr_LIF_FAAM['LIF_tm_final']*slope + intercept), color='red', label='Theil-Sen')
plt.legend()
plt.xlabel('LIF SO2 (ppb)')
plt.ylabel('PF SO2 (ppb)')
Corr_LIF_FAAM.index = Corr_LIF_FAAM.index.tz_localize(None)
#Corr_LIF_FAAM.to_excel('Correlation_LIF_FAAM_400ppt.xlsx')




### LIF-CIMS (CIMS in cps)
Corr_LIF_CIMS = pd.DataFrame([LIF_CIMS_all_concat['LIF_tm_final'], CIMS_LIF_all_concat['CIMS_tm_final']])
Corr_LIF_CIMS = Corr_LIF_CIMS.transpose()

# for i in Corr_LIF_CIMS.index:
#     if Corr_LIF_CIMS['CIMS_tm_final'][i] < 50:
#         Corr_LIF_CIMS['LIF_tm_final'][i] = np.nan
#         Corr_LIF_CIMS['CIMS_tm_final'][i] = np.nan

for i in Corr_LIF_CIMS.index:
    if Corr_LIF_CIMS['LIF_tm_final'][i] < 0.07:
        Corr_LIF_CIMS['LIF_tm_final'][i] = np.nan
        Corr_LIF_CIMS['CIMS_tm_final'][i] = np.nan
        
Corr_LIF_CIMS['LIF_tm_final_err'] = Corr_LIF_CIMS['LIF_tm_final'] * (10/100) + 0.0065
Corr_LIF_CIMS['CIMS_tm_final_err'] = Corr_LIF_CIMS['CIMS_tm_final'] * (102/100)
Corr_LIF_CIMS = Corr_LIF_CIMS.dropna()

# Using York regression
from York_regression_script import York_reg
b, b_err, a, a_err, stat1, stat2 = York_reg(x=Corr_LIF_CIMS['LIF_tm_final'], y=Corr_LIF_CIMS['CIMS_tm_final'],
         sx=Corr_LIF_CIMS['LIF_tm_final_err'], sy=Corr_LIF_CIMS['CIMS_tm_final_err'],
         Ri=0, b0=12)

# Using orthogonal distance regression
from scipy.odr import ODR, Model, RealData
def f(beta, x):
    m, p = beta
    return m*x + p
data = RealData(x=Corr_LIF_CIMS['LIF_tm_final'], y=Corr_LIF_CIMS['CIMS_tm_final'],
         sx=Corr_LIF_CIMS['LIF_tm_final_err'], sy=Corr_LIF_CIMS['CIMS_tm_final_err'])
model = Model(f)
odr = ODR(data, model, beta0=[1., 0.])
output = odr.run()
output.pprint()


# Using OLS
def func(x, a, b):
    return a * x + b
#scipy.optimize.curve_fit(func, Corr_both['LIF_tm_final'], Corr_both['CIMS_tm_final'])
popt, pcov = scipy.optimize.curve_fit(func, Corr_LIF_CIMS['LIF_tm_final'], Corr_LIF_CIMS['CIMS_tm_final'])
residuals = Corr_LIF_CIMS['CIMS_tm_final']- func(Corr_LIF_CIMS['LIF_tm_final'], *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Corr_LIF_CIMS['CIMS_tm_final']-np.mean(Corr_LIF_CIMS['CIMS_tm_final']))**2)
r_squared = 1 - (ss_res / ss_tot)

# Using Theil-Sen
from scipy.stats import theilslopes

slope, intercept, lo_slope, hi_slope = theilslopes(Corr_LIF_CIMS['CIMS_tm_final'], Corr_LIF_CIMS['LIF_tm_final'],
                                                   alpha=0.95)

        
plt.scatter(Corr_LIF_CIMS['LIF_tm_final'], Corr_LIF_CIMS['CIMS_tm_final'])
plt.plot(Corr_LIF_CIMS['LIF_tm_final'], (Corr_LIF_CIMS['LIF_tm_final']*popt[0] + popt[1]))
plt.plot(Corr_LIF_CIMS['LIF_tm_final'], (Corr_LIF_CIMS['LIF_tm_final']*b + a), color='orange', label='York')
plt.plot(Corr_LIF_CIMS['LIF_tm_final'], (Corr_LIF_CIMS['LIF_tm_final']*output.beta[0] + output.beta[1]), color='green', label='ODR')
plt.plot(Corr_LIF_CIMS['LIF_tm_final'], (Corr_LIF_CIMS['LIF_tm_final']*slope + intercept), color='red', label='Theil-Sen')
plt.legend()
plt.xlabel('LIF SO2 (ppb)')
plt.ylabel('CIMS SO2 (ppb)')
Corr_LIF_CIMS.index = Corr_LIF_CIMS.index.tz_localize(None)
#Corr_LIF_CIMS.to_excel('Correlation_LIF_CIMS_70ppt.xlsx')






#York regression code
x = np.array(Corr_LIF_FAAM['LIF_tm_final'])
y = np.array(Corr_LIF_FAAM['FAAM_tm_final'])
sx = np.array(0.1*Corr_LIF_FAAM['LIF_tm_final'])
sy = np.array(0.18*Corr_LIF_FAAM['FAAM_tm_final'])
b0=1
Ri=-0
eps=1e-8
#tol = 1e-7
tol = abs(b0) * eps # the fit will stop iterating when the slope converges to within this value

wx = 1 / (sx**2) # weight x
wy = 1 / (sy**2) # weight y

# York regression - iterative calculation of slope and intercept
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

# Error calculation
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

# Goodness of fit calculation
lgth = len(x)
wSint = Wi * (y - b*x - a)**2
wSint = wSint[~np.isnan(wSint)]
sumSint = sum(wSint)

# Plot scatter points and trendline
fig, ax = plt.subplots()
ax.errorbar(x, y, sy, sx, marker='o', linestyle='')
ax.set_xlabel('SO$_2$ Mixing Ratio (ppt)', fontsize=14)
ax.set_ylabel('LIF Signal (cps/mW)', fontsize=14)
yp = b*x + a
ax.plot(x, yp, linestyle='dashed')
ax.tick_params(axis='both', labelsize=12)
ax.annotate('sensitivity = %.1f +/- %.1f\nintercept = %.0f +/- %.0f' % (b, b_err, a, a_err), xycoords='axes fraction', xy=(0.05,0.9), fontsize=12)

