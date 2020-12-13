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

patches, sizes = patches.patches_from_region_list(region_list)

francais = True

data_patches = gather_data(patches, SOS_medecins=False, include_early=False)

deces = 'Décès hospitaliers dus au Covid-19'
admissions = 'Admissions à l\'hôpital liées au Covid-19'

# data_patches[0] = data_patches[0].drop(['ICU admissions'], axis = 1)
for i in range(len(names)):
    data_patches[i].columns = [admissions, deces, 'Admissions en réanimation']

fit_total = lockdown.FitPatches(data_patches, names, sizes)

## sans second confinement
fit_total.dates_of_change = ['2020-03-16', '2020-06-02', '2020-07-10', '2020-09-01']
fit_total.start_fit_init = '2020-03-19'
fit_total.end_fit_init = '2020-03-26'
fit_total.dates_end_fit = ['2020-06-10', '2020-07-29', '2020-09-12', '2020-10-30']
fit_total.names_fit = ['confinement', 'après le 2 juin', 'été', 'deuxieme vague']
fit_total.delays = np.array([[18, 28, 28], [10,15], [18, 21, 21], [10, 15, 15]])
fit_total.fit_columns = [None, [admissions, deces], None, None] # None means all columns

fit_total.fit_patches(reference_column = deces)

fit_total.plot_fit_lockdown(francais = francais)

deaths = []
times = []
for i in range(fit_total.n):
    deaths.append(pd.DataFrame())
    times.append(pd.DataFrame())

ifrs = [0.003, 0.005, 0.01]
for f in ifrs:
    fit_total.prepare_sir(.8, f, ref_event = deces)
    fit_total.adjust_dates_of_change('été', admissions)
    fit_total.adjust_dates_of_change('deuxieme vague', admissions)
    fit_total.compute_sir(.8, f, '2021-03-31', ref_event = deces)
    
    for (j, sir) in enumerate(fit_total.sir):
        deaths[j][str(f)] = pd.Series(sir.daily[deces])
        times[j][str(f)] = pd.Series(sir.times-sir.lockdown_time)
    

    
fig, axs = plt.subplots(2,2, dpi = 200, figsize=(12,6))

for (j, patch) in enumerate(fit_total.names):
    k = np.mod(j, 2)
    i = int(np.floor(j/2))
    dates = fit_total.fitters[j].daily.index
    timesindex = fit_total.index_to_time(dates)
    axs[i,k].set_title(patch)
    axs[i,k].grid(True)
    axs[i,k].plot(timesindex, fit_total.fitters[j].daily[deces].values, linestyle ='dashed')
    for f in ifrs:
        axs[i,k].plot(times[j][str(f)].values, deaths[j][str(f)].values)
    start = np.min(np.min(times[j]))
    end = np.max(np.max(times[j]))
    ticks = np.linspace(start, end, 5)
    dates = fit_total.time_to_date(ticks)
    axs[i,k].set_xticks(ticks)
    axs[i,k].set_xticklabels(dates)

fig.set_tight_layout(True)



# fit_total.prepare_sir(.8, .005, ref_event = deces)
# fit_total.adjust_dates_of_change('été', admissions)
# fit_total.adjust_dates_of_change('deuxieme vague', admissions)

# fit_total.compute_sir(.8, .005, '2021-03-31', ref_event = deces)

# fit_total.plot_events(logscale=False)
# fit_total.axs[0].set_title("Projection de l'évolution de l'épidémie sans le second confinement")