#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:22:37 2020

@author: raphael
"""

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np

from gather_data import gather_data
import build_patches
import fit_lockdown as lockdown
import datetime as date

region_list = [['Île de France']]
names = ['Île de France']


event = 'Hospital deaths'

CHANGE_INPUT = True
COMPUTE_AGAIN = True

class Change(Exception):
    pass

inputs = {
    'num_vars': 4,
    'names': ['exposed period', 'reported proportion', 
              'lockdown date', 'infection fatality ratio'],
    'bounds': [[2, 5],
               [.2, .8],
               [-3,3],
               [.003, .01]]}

try:
    if CHANGE_INPUT:
        raise Change('input')
    param_values = np.loadtxt("param_values_delays.txt", float)
except (OSError, Change):
    param_values = saltelli.sample(inputs, 3)
    np.savetxt("param_values_delays.txt", param_values)

try:
    if COMPUTE_AGAIN:
        raise Change('output')
    output_values = np.loadtxt('output_values_delays.txt', float)
except (OSError, Change):
    output_values = np.zeros((2, np.size(param_values, 0)))
    
    ## prepare lockdown fitter
    patches, sizes = build_patches.patches_from_region_list(region_list)
    data_patches = gather_data(patches, SOS_medecins=False, include_early=False)
    fitter = lockdown.LockdownFitter(data_patches[0], names[0], sizes[0], '2020-03-16')
    fitter.setup_fit('Before lockdown', '2020-03-01', '2020-03-26', [17], columns = ['Hospital deaths'])
    fitter.setup_fit('Lockdown', '2020-03-16', '2020-05-11', [18, 28, 28])
    fitter.compute_growth_rates()
    
    for (i, param) in enumerate(param_values):
        E_dist, I_dist = lockdown.EI_dist(param[0], param[1], 10, n=30)
        EI_dist = lockdown.product_dist(E_dist, I_dist)
        lockdown_datetime = fitter.datetime_lockdown + date.timedelta(days = param[2])
        fitter.datetime_lockdown = lockdown_datetime
        fitter.prepare_sir(EI_dist, param[3], verbose = False)
        output_values[0,i] = fitter.param_delays[event].product()
        fitter.compute_sir(EI_dist, param[3], '2020-05-11', verbose = False, compute_events = False)
        output_values[1,i] = 1-fitter.sir.Z[0]
        
    np.savetxt('output_values_delays.txt', output_values)
    
Sobol_indices = sobol.analyze(inputs, output_values, print_to_console = True)