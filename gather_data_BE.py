#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:31:27 2020

@author: raphael
"""

import pandas as pd
import numpy as np
import os as os

import fit_lockdown as lockdown

data_deaths_file = 'COVID19BE_MORT.csv'
data_hosp_file = 'COVID19BE_HOSP.csv'

def gather_data_BE():
    deaths_data = pd.read_csv(data_deaths_file)
    deaths_data = deaths_data.groupby('DATE').sum()
    deaths_data = deaths_data.cumsum()
    
    hosp_data = pd.read_csv(data_hosp_file)
    hosp_data = hosp_data.groupby('DATE').sum()
    admissions = hosp_data['TOTAL_IN']['2020-03-15'] + hosp_data['NEW_IN'].cumsum()
    
    data = pd.DataFrame()
    
    data['Hospital deaths'] = deaths_data['DEATHS']
    data['Hospital admissions'] = admissions
    return data

data = [gather_data_BE()]
# print(data)

fit_BE = lockdown.FitPatches(data, ['Belgium'], [11431406])

fit_BE.lockdown_date = '2020-03-17'
fit_BE.delays = np.array([[28, 18], [15, 10], [15, 10], [28, 18]])
fit_BE.start_fit_init = '2020-03-11'
fit_BE.end_fit_init = '2020-03-21'
fit_BE.dates_of_change = ['2020-03-17', '2020-05-11', '2020-06-15', '2020-08-05']
fit_BE.dates_end_fit = ['2020-05-11', '2020-06-15', '2020-08-14', '2020-10-04']

fit_BE.fit_patches()

fit_BE.plot_fit_lockdown()

fit_BE.compute_sir(.8, .006, '2020-10-31', Markov = False)
fit_BE.plot_events()
fit_BE.sir[0].plot(S = False)
fit_BE.sir[0].ax.plot(fit_BE.sir[0].times, 1-fit_BE.sir[0].traj[:,0], label = '1-S')
fit_BE.sir[0].ax.legend(loc='best')
