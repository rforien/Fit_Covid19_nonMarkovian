#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:08:28 2020

@author: raphael
"""

import pandas as pd
import numpy as np
import os as os

data_hosp_file = 'donnees-hospitalieres-covid19-2020-07-13-19h00_corrected.csv'
data_hosp_daily_file = 'donnees-hospitalieres-nouveaux-covid19-2020-07-13-19h00.csv'
data_sos_file = 'sursaud-corona-quot-dep-2020-07-13-19h15_corrected.csv'

columns = ['Hospital admissions', 'Hospital deaths', 'ICU admissions', 'SOS Medecins actions']

def correct_hosp_data():
    data = pd.read_csv(data_hosp_file[:-14] + '.csv', delimiter = ';')

    # correct date format
    dates = ['27/06/2020', '28/06/2020', '29/06/2020']
    correct_dates = ['2020-06-27', '2020-06-28', '2020-06-29']
    
    for (i, date) in enumerate(dates):
        index = np.where(data['jour'] == date)[0]
        for j in index:
    #        print(j)
            # print(data['dep'].values[j], data['jour'].values[j])
            data['jour'][j] = correct_dates[i]
    print(np.unique(data['jour']))
    
    # remove duplicate entries for 2020-07-03
    index = np.where(data['jour'] == '2020-07-03')[0]
    if np.size(index) == 6*101:
        print('remove duplicate')
        k = int(np.size(index)/2)
        data = data.drop(index[k:])
    
    data.to_csv(data_hosp_file, sep = ';')

def correct_sos_data():
    data_sos = pd.read_csv(data_sos_file[:-14] + '.csv', delimiter = ';')

    index = np.where(data_sos['dep'] == 40)[0]
    for j in index:
        data_sos['dep'][j] = '40'
    
    print(np.sum(data_sos['dep'] == 40))
    
    data_sos.to_csv(data_sos_file, sep = ';')

def gather_data(patches, include_early = False, index = 0):
    if not os.path.isfile(data_hosp_file):
        correct_hosp_data()
    if not os.path.isfile(data_sos_file):
        correct_sos_data()
    
    data_hosp = pd.read_csv(data_hosp_file, delimiter = ';')
    data_hosp_daily = pd.read_csv(data_hosp_daily_file, delimiter = ';')
    data_sos = pd.read_csv(data_sos_file, delimiter = ';')
    
    if include_early:
        deaths_early = pd.read_csv('deces_france_0101-1404.csv', index_col = 'jour')
    
    # forget sex
    data_hosp = data_hosp[data_hosp['sexe'] == 0]
    data_sos = data_sos[data_sos['sursaud_cl_age_corona'] == '0']
    # dep as string
    data_sos['dep'] = [str(dep) for dep in data_sos['dep'].values]
    
    deaths = data_hosp.pivot(index = 'jour', columns = 'dep', values = 'dc')
    data_hosp['cumul_hosp'] = data_hosp['hosp'] + data_hosp['rad'] + data_hosp['dc']
    admissions = data_hosp.pivot(index = 'jour', columns = 'dep', values = 'cumul_hosp')
    rea = data_hosp_daily.pivot(index = 'jour', columns = 'dep', values = 'incid_rea').cumsum()
    in_rea = data_hosp.pivot(index = 'jour', columns = 'dep', values = 'rea')
    for dep in rea.columns:
        rea[dep] += in_rea[dep].values[0]
    actes_sos = data_sos.pivot(index = 'date_de_passage', columns = 'dep', values = 'nbre_acte_corona').cumsum()
    
    n = np.size(patches)
    data_patches = n*[pd.DataFrame]
    for (i, patch) in enumerate(patches):
        deaths_patch = deaths[patch].sum(axis = 1)
        if include_early and i == index:
            deaths_patch = pd.concat((deaths_early['2020-02-15':'2020-03-17'], deaths_patch), axis = 0)
        admis_patch = admissions[patch].sum(axis = 1)
        rea_patch = rea[patch].sum(axis = 1)
        sos_patch = actes_sos[patch].sum(axis = 1)
        data_patches[i] = pd.concat((admis_patch, deaths_patch, rea_patch, sos_patch), axis = 1)
        data_patches[i].columns = columns
        data_patches[i].sort_index(inplace = True)
    
    return data_patches
