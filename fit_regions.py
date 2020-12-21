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
            fitter.adjust_date_of_change(fit.name, 'Hospital admissions')
        fitter.compute_sir('2020-12-31')
        
        
        fitter.plot_fit()
        fitter.plot_delays()
        fitter.sir.plot()
        fitter.plot_events()
    except Exception as error:
        print('Error !!!!!!')
        print(error)
        print(traceback.format_exc())
    finally:
        return fitter


if __name__ == '__main__':
    fitter = fit_IDF(.8, .005)