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
print(data)

fit_BE = lockdown.FitPatches(data, ['Belgium'], [11431406])

fit_BE.delays = np.array([[28, 18], [15, 10], [15, 10], [28, 18]])
fit_BE.start_fit_init = '2020-03-10'

fit_BE.fit_patches()

fit_BE.plot_fit_lockdown()
