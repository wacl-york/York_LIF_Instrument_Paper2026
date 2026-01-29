Data and code to reproduce the figures in the publication 'An intercomparison of aircraft sulfur dioxide measurements in clean and polluted marine environments' by L. G. Temple et al. (2026)


ACRUISE-3 Data

Each flight contains a 'CXXX.py' and 'CXXX_functions.py' script which:
- Reads in LIF Signal data (processed data_050224_sig and processed data_050224_ref)
- Reads in LIF sensitivity data (All_sensitivities_030924.xlsx, and split into calibrations in ambient air: 'Sensitivities_amb_030924.xlsx' and calibrations in zero air: 'Sensitivities_za_030924.xlsx')
  and displays sensitivity with time
- Applies mean LIF sensitivity to LIF Signal data and removes calibration times using 'All_cal_times.xlsx'
- Averages LIF data, including to 10 s (find York_LIF_SO2_mr_10s_CXXX.csv as an example output)
- Reads in FAAM PF SO2 and altitude data from the 'PF_data' folder
- Reads in CIMS SO2 data from the 'CIMS_data' folder
- Reads in the FAAM CO2 data from the 'CO2_data' folder
- Code to produce various plots in the publication.

Each flight contains 'Plume_int_CXXX.py' which:
- Reads in data outputted from CXXX.py to calculate emission ratios and associated uncertainties via the a) integration method using http://github.com/wacl-york/acruise-peakid and
  b) regression method

The ACRUISE-3 folder also contains 'Correlation_concat.py' for producing plots in Fig. 8 using data from all three flights (outputted from CXXX.py) and 
LIF mixing ratios binned by altitude (ACRUISE_SO2_mr_alt_mean_median.xlsx) in Fig. S9.

The FAAM PF SO2, CO2 and altitude data for ACRUISE-3 can also be found here:
Facility for Airborne Atmospheric Measurements; Natural Environment Research Council; Met Office (2022): FAAM C285, C286, C287 ACRUISE flights: 
Airborne atmospheric measurements from core instrument suite on board the BAE-146 aircraft. NERC EDS Centre for Environmental Data Analysis, 
last access: 28/01/26. https://catalogue.ceda.ac.uk/uuid/d6eb4e907c124482881d7d03c06903e4/



ACSIS-7 Data

Each flight contains a 'CXXX.py' and 'CXXX_functions.py' script which:
- Reads in LIF Signal data (processed data_030723_sig and processed data_030723_ref)
- Reads in LIF sensitivity data (All_sensitivities_030723.xlsx, and split into calibrations in ambient air: 'Sensitivities_amb_030723.xlsx' and calibrations in zero air: 'Sensitivities_za_030723.xlsx')
and displays sensitivity with time
- Applies mean LIF sensitivity to LIF Signal data and removes calibration times using 'All_cal_times.xlsx'
- Averages LIF data, including to 10 s (find York_LIF_SO2_mr_10s_CXXX.csv as an example output)
- Reads in FAAM PF SO2 and altitude data from the 'PF_data' folder

The ACSIS-7 folder also contains 'York_LIF_SO2_Alt.py' for producing LIF mixing ratios binned by altitude (ACSIS_SO2_mr_alt_mean_median.xlsx) in Fig. S9.

The FAAM PF SO2 and altitude data for ACSIS-7 can also be found here:
Facility for Airborne Atmospheric Measurements; Natural Environment Research Council; Met Office (2022): FAAM C289, C290, C292, C293 ACSIS flights: 
Airborne atmospheric measurements from core instrument suite on board the BAE-146 aircraft. NERC EDS Centre for Environmental Data Analysis, 
last access: 28/01/26. https://catalogue.ceda.ac.uk/uuid/7e92f3a40afc494f9aaf92525ebb4779 
