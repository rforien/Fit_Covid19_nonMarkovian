#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:35:59 2020

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt

from gather_data import gather_data
import build_patches
import fit_lockdown as lockdown

region_list = [['Île de France']]
names = ['Île de France']

# region_list = [['Île de France'], ["Provence-Alpes-Côte d’Azur"]]
# names = ['Ile de France', "Provence-Alpes-Cote d'Azur"]

patches, sizes = build_patches.patches_from_region_list(region_list)

francais= True

data_patches = gather_data(patches, SOS_medecins=False, include_early=False)

fitters = []
for (i, name) in enumerate(names):
    fitters.append(lockdown.LockdownFitter(data_patches[i], name, sizes[i], '2020-03-16'))
    fitters[i].setup_fit('Before lockdown', '2020-03-01', '2020-03-26', [17], columns = ['Hospital deaths'])
    fitters[i].setup_fit('Lockdown', '2020-03-16', '2020-05-11', [18, 28, 28])
    fitters[i].compute_growth_rates()
    EI_dist = lockdown.EI_dist_covid(.8)
    fitters[i].prepare_sir(EI_dist, 0.005)
    fitters[i].compute_sir(EI_dist, .005, '2020-06-06')

for fitter in fitters:
    fitter.plot_fit()
    fitter.plot_delays()
    fitter.sir.plot()
    fitter.plot_events()