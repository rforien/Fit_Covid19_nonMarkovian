#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:26:30 2020

@author: raphael
""" 

import numpy as np
import matplotlib.pyplot as plt
import traceback

from gather_data import gather_hosp_data
import build_patches
import fit_lockdown as lockdown

hosp_file = 'donnees-hospitalieres-covid19-2020-12-19-19h03_corrected.csv'
hosp_daily_file = 'donnees-hospitalieres-nouveaux-covid19-2020-12-19-19h03.csv'

def fit_IDF(p_reported, ifr):
    name = ['Île de France']
    departements, size = build_patches.patches_from_regions(name)
    data = gather_hosp_data(departements[0], hosp_file, hosp_daily_file)
    fitter = lockdown.LockdownFitter(data, name[0], size[0], '2020-03-16')
    fitter.setup_fit('Before lockdown', '2020-03-01', '2020-03-27', [17], 
                     columns = ['Hospital deaths'])
    fitter.setup_fit('Lockdown', '2020-03-16', '2020-05-11', [18, 28, 28])
    fitter.setup_fit('Easing of lockdown', '2020-05-11', '2020-06-10',
                     [10, 15, 15])
    fitter.setup_fit('Second wave', '2020-07-10', '2020-10-30',
                     [18, 28, 28])
    fitter.setup_fit('Second lockdown', '2020-10-30', '2020-12-01',
                     [13, 18, 18])
    fitter.setup_fit('Easing of second lockdown', '2020-11-25', '2020-12-18',
                     [10, 15, 15])
    fitter.compute_growth_rates(verbose = True)
    
    try:
        EI_dist = lockdown.EI_dist_covid(p_reported)
        fitter.prepare_sir(EI_dist, ifr)
        for fit in fitter.fits[2:]:
            if fit.name == 'Second lockdown':
                continue
            fitter.adjust_date_of_change(fit.name, 'Hospital admissions')
        fitter.compute_sir('2020-12-31')
        
    except Exception as error:
        print('Error !!!!!!')
        print(error)
        print(traceback.format_exc())
    finally:
        return fitter

def fit_GrandEst(p_reported, ifr, curfew = True):
    name = ['Grand Est']
    departements, size = build_patches.patches_from_regions(name)
    data = gather_hosp_data(departements[0], hosp_file, hosp_daily_file)
    fitter = lockdown.LockdownFitter(data, name[0], size[0], '2020-03-16')
    fitter.setup_fit('Before lockdown', '2020-03-01', '2020-03-25', [17], 
                     columns = ['Hospital deaths'])
    fitter.setup_fit('Lockdown', '2020-03-16', '2020-05-11', [18, 28, 28])
    fitter.setup_fit('Easing of lockdown', '2020-05-11', '2020-06-10',
                      [10, 15, 15])
    fitter.setup_fit('Second wave (1/2)', '2020-07-10', '2020-10-01',
                      [18, 28, 28])
    fitter.setup_fit('Second wave (2/2)', '2020-09-25', '2020-10-30',
                     [15, 21, 21])
    if curfew:
        fitter.setup_fit('Curfew', '2020-10-23', '2020-11-10',
                         [8], ['Hospital admissions'])
    fitter.setup_fit('Second lockdown', '2020-10-30', '2020-12-01',
                      [13, 18], ['Hospital admissions', 'ICU admissions'])
    fitter.setup_fit('Easing of second lockdown', '2020-11-25', '2020-12-18',
                      [10, 15, 15])
    fitter.compute_growth_rates(verbose = True)
    
    try:
        EI_dist = lockdown.EI_dist_covid(p_reported)
        fitter.prepare_sir(EI_dist, ifr)
        for fit in fitter.fits[2:]:
            if fit.name == 'Second lockdown' or fit.name == 'Curfew':
                continue
            fitter.adjust_date_of_change(fit.name, 'Hospital admissions')
        fitter.compute_sir('2020-12-31')
        
    except Exception as error:
        print('Error !!!!!!')
        print(error)
        print(traceback.format_exc())
    finally:
        return fitter
    

def fit_PACA(p_reported, ifr, curfew = True):
    name = ["Provence-Alpes-Côte d’Azur"]
    departements, size = build_patches.patches_from_regions(name)
    data = gather_hosp_data(departements[0], hosp_file, hosp_daily_file)
    fitter = lockdown.LockdownFitter(data, name[0], size[0], '2020-03-16')
    fitter.setup_fit('Before lockdown', '2020-03-01', '2020-03-27', [17], 
                     columns = ['Hospital deaths'])
    fitter.setup_fit('Lockdown', '2020-03-16', '2020-05-11', [18, 28, 28])
    fitter.setup_fit('Easing of lockdown', '2020-05-01', '2020-06-20',
                      [10, 15], ['Hospital admissions', 'Hospital deaths'])
    fitter.setup_fit('Second wave (1/3)', '2020-06-28', '2020-09-07',
                      [18, 48, 32])
    fitter.setup_fit('Second wave (2/3)', '2020-09-01', '2020-10-10',
                      [15, 18, 18])
    fitter.setup_fit('Second wave (3/3)', '2020-09-25', '2020-10-20',
                     [15, 18, 18])
    if curfew:
        fitter.setup_fit('Curfew', '2020-10-16', '2020-11-10',
                         [8], ['Hospital admissions'])
    fitter.setup_fit('Second lockdown', '2020-10-30', '2020-12-01',
                      [13, 15, 15])
    fitter.setup_fit('Easing of second lockdown', '2020-11-29', '2020-12-18',
                      [10, 15, 15])
    fitter.compute_growth_rates(verbose = True)
    
    try:
        EI_dist = lockdown.EI_dist_covid(p_reported)
        fitter.prepare_sir(EI_dist, ifr)
        for fit in fitter.fits[2:]:
            if fit.name in ['Second lockdown', 'Curfew', 'Easing of second lockdown']:
                continue
            fitter.adjust_date_of_change(fit.name, 'Hospital admissions')
        fitter.compute_sir('2020-12-31')
        
    except Exception as error:
        print('Error !!!!!!')
        print(error)
        print(traceback.format_exc())
    finally:
        return fitter


def fit_Auvergne_Rhone_Alpes(p_reported, ifr):
    name = ["Auvergne-Rhône-Alpe"]
    departements, size = build_patches.patches_from_regions(name)
    data = gather_hosp_data(departements[0], hosp_file, hosp_daily_file)
    fitter = lockdown.LockdownFitter(data, name[0], size[0], '2020-03-16')
    fitter.setup_fit('Before lockdown', '2020-03-15', '2020-03-28', 
                     [3], columns = ['Hospital deaths'])
    fitter.setup_fit('Lockdown', '2020-03-16', '2020-05-11', [18, 28, 28])
    fitter.setup_fit('Easing of lockdown', '2020-05-11', '2020-06-20',
                      [10, 15, 15])
    fitter.setup_fit('Second wave', '2020-06-28', '2020-10-30',
                      [18, 48, 37])
    # fitter.setup_fit('Curfew', '2020-10-16', '2020-11-10',
    #                   [8], ['Hospital admissions'])
    fitter.setup_fit('Second lockdown', '2020-10-30', '2020-12-01',
                      [13, 21, 15])
    fitter.setup_fit('Easing of second lockdown', '2020-11-29', '2020-12-18',
                      [10, 15, 15])
    fitter.compute_growth_rates(verbose = True)
    
    try:
        EI_dist = lockdown.EI_dist_covid(p_reported)
        fitter.prepare_sir(EI_dist, ifr)
        for fit in fitter.fits[2:]:
            if fit.name in ['Second lockdown', 'Curfew', 'Easing of second lockdown']:
                continue
            fitter.adjust_date_of_change(fit.name, 'Hospital admissions')
        fitter.compute_sir('2020-12-31')
        
    except Exception as error:
        print('Error !!!!!!')
        print(error)
        print(traceback.format_exc())
    finally:
        return fitter

if __name__ == '__main__':
    p_reported = .8
    ifr = .005
    fitters = []
    fitters.append(fit_IDF(p_reported, ifr))
    fitters.append(fit_GrandEst(p_reported, ifr))
    fitters.append(fit_PACA(p_reported, ifr))
    fitters.append(fit_Auvergne_Rhone_Alpes(p_reported, ifr))
    
    n = len(fitters)
    l = int(np.floor(n/2))
    k = int(np.ceil(n/l))
    
    fig1, axes1 = plt.subplots(l,k, dpi = 200, figsize=(12,9))
    for i in range(k):
        for j in range(l):
            if l*i+j >= n:
                continue
            data_lines = fitters[l*i+j].plot_fit(axs = axes1[j,i], legend = True, nb_xticks = 4)
    fig1.legend(data_lines, fitters[0].events, loc=(.8, .05), 
                fontsize = 12, framealpha = 1)
    plt.tight_layout(True, rect = (0, 0, 1, 1))
    
    
    fig2, axes2 = plt.subplots(l,k, dpi = 200, figsize=(12,9))
    for i in range(k):
        for j in range(l):
            if l*i+j >= n:
                continue
            data_lines, sir_lines = fitters[l*i+j].plot_events(axs = axes2[j,i], nb_xticks = 4)
    predic_labels = 3*['Model predictions']
    fig2.legend(tuple(data_lines + sir_lines), tuple(fitters[0].events) + tuple(predic_labels), 
                loc=(.09, .85), fontsize = 12, ncol = 2)
    plt.tight_layout(True)