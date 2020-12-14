#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:08:49 2020

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gather_data import gather_data
import build_patches as patches
import fit_lockdown as lockdown

region_list = [['Île de France'], ['Grand Est'], ['Nouvelle Aquitaine'], ["Provence-Alpes-Côte d’Azur"]]
names = ['Île de France', 'Grand Est', 'Nouvelle Aquitaine', "Provence-Alpes-Côte d’Azur"]
idf = 0
gdest = 1
nvaq = 2
paca = 3

patches, sizes = patches.patches_from_region_list(region_list)

francais = True

data_patches = gather_data(patches, SOS_medecins=False, include_early=False)

deces = 'Décès hospitaliers dus au Covid-19'
admissions = 'Admissions à l\'hôpital liées au Covid-19'

# data_patches[0] = data_patches[0].drop(['ICU admissions'], axis = 1)
for i in range(len(names)):
    data_patches[i].columns = [admissions, deces, 'Admissions en réanimation']

patches = []
for (i, name) in enumerate(names):
    data_patches[i] = data_patches[i].drop(data_patches[i]['2020-11-01':].index)
    patches.append(lockdown.LockdownFitter(data_patches[i], name, sizes[i], '2020-03-16'))

for patch in patches:
    patch.setup_fit('avant le confinement', '2020-03-01', '2020-03-25', [17], columns = [deces])
    patch.setup_fit('confinement', '2020-03-16', '2020-06-02', [18, 28, 28])
    
## Setup fits in Ile de France
patches[idf].setup_fit('été', '2020-06-02', '2020-09-10', [18, 25, 25])
patches[idf].setup_fit('deuxième vague', '2020-09-01', '2020-10-30', [10, 15, 15])

## Setup fits in Grand Est
patches[gdest].setup_fit('déconfinement', '2020-05-15', '2020-07-01', [15, 20, 20])
patches[gdest].setup_fit('été', '2020-07-01', '2020-09-20', [15, 20, 20])
patches[gdest].setup_fit('deuxième vague', '2020-09-01', '2020-10-30', [10, 15, 15])

## Setup fits in Nouvelle Aquitaine
# patches[nvaq].setup_fit('déconfinement', '2020-06-02', '2020-08-01', [10], columns = [admissions])
patches[nvaq].setup_fit('deuxième vague', '2020-07-25', '2020-10-30', [15, 20, 28])

## Setup fits in PACA
patches[paca].setup_fit('deuxième vague', '2020-07-01', '2020-10-30', [15, 28, 20])

for patch in patches:
    patch.compute_growth_rates()

fig, axes = plt.subplots(2,2, dpi = 200, figsize=(12,9))
for i in range(2):
    for j in range(2):
        patches[2*i+j].plot_fit(axs = axes[i,j], francais = True, nb_xticks = 4)
fig.set_tight_layout(True)
    
# ifrs = [0.003, 0.005, 0.01]
ifrs = [.003, .005]
EI_dist = lockdown.EI_dist_covid(0.8)

deaths = pd.DataFrame()
times = pd.DataFrame()

for f in ifrs:
    for patch in patches:
        if not hasattr(patch, 'sir'):
            patch.prepare_sir(EI_dist, f, ref_event = deces)
        else:
            patch.sir.forget()
            patch.compute_probas(f, ref_event = deces)
        for (i, fit) in enumerate(patch.fits):
            if i <= 1:
                continue
            if fit.name == 'deuxième vague' and patch.name == 'Grand Est':
                patch.adjust_date_of_change(fit.name, deces)
            else:
                patch.adjust_date_of_change(fit.name, admissions)
        patch.compute_sir(EI_dist, f, '2020-02-27', ref_event = deces)
        
        deaths[patch.name + str(f)] = pd.Series(patch.sir.daily[deces])
        times[patch.name + str(f)] = pd.Series(patch.sir.times-patch.sir.lockdown_time)
    


fig, axs = plt.subplots(2,2, dpi = 200, figsize=(12,9))
for i in range(2):
    for j in range(2):
        patch = patches[2*i+j]
        dates = patch.fitter.daily.index
        timesindex = patch.date_to_time(dates)
        axs[i,j].set_title(patch.name)
        axs[i,j].grid(True)
        axs[i,j].plot(timesindex, patch.fitter.daily[deces].values, linestyle ='dashed')
        for f in ifrs:
            axs[i,j].plot(times[patch.name + str(f)].values, 
                          deaths[patch.name + str(f)].values, label = "IFR = %.1f%%" % (100*f))
        start = np.min(np.min(times[patch.name + str(ifrs[0])]))
        end = np.max(np.max(times[patch.name + str(ifrs[0])]))
        ticks = np.linspace(start, end, 5)
        tick_dates = patch.time_to_date(ticks)
        axs[i,j].set_xticks(ticks)
        axs[i,j].set_xticklabels(tick_dates)

axs[0,0].legend(loc='best')
fig.set_tight_layout(True)

# deaths = []
# times = []
# for i in range(fit_total.n):
#     deaths.append(pd.DataFrame())
#     times.append(pd.DataFrame())


