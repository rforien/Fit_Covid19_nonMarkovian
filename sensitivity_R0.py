#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:22:27 2020

@author: raphael
"""

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np

import fit_lockdown as lockdown

CHANGE_INPUT = True
COMPUTE_AGAIN = True

class Change(Exception):
    pass

inputs = {
    'num_vars': 3,
    'names': ['exposed period', 'reported proportion', 'max infectious period'],
    'bounds': [[2, 5],
               [.2, .8],
               [8, 12]]}

rho = np.log(2)/16

try:
    if CHANGE_INPUT:
        raise Change('input')
    param_values = np.loadtxt("param_values_R0.txt", float)
except (OSError, Change):
    param_values = saltelli.sample(inputs, 10000)
    np.savetxt("param_values_R0.txt", param_values)

try:
    if COMPUTE_AGAIN:
        raise Change('output')
    output_values = np.loadtxt('output_values_R0.txt', float)
except (OSError, Change):
    output_values = np.zeros(np.size(param_values, 0))
    
    for (i, param) in enumerate(param_values):
        E_dist, I_dist = lockdown.EI_dist(param[0], param[1], param[2], n=30)
        MGF_E = lockdown.Laplace(E_dist, -rho)
        MGF_I = lockdown.Laplace(I_dist, -rho)
        R0 = rho*np.sum(I_dist[:,0]*I_dist[:,1])/(MGF_E*(1-MGF_I))
        output_values[i] = R0
    np.savetxt('output_values_R0.txt', output_values)
    
Sobol_indices = sobol.analyze(inputs, output_values, print_to_console = True)