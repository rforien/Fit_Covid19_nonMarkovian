#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 22:00:25 2020

@author: raphael
"""

import numpy as np

from gather_data import gather_data
import build_patches as patches
import fit_lockdown as lockdown

region_list = [['Île de France', 'Grand Est', 'Hauts-de-France', 'Bourgogne-Franche-Comté', 
                'Auvergne-Rhône-Alpe', "Provence-Alpes-Côte d’Azur",
                'Nouvelle Aquitaine', 'Occitanie',
                'Bretagne', 'Centre-Val de Loire', 'Normandie', 'Pays de la Loire']]
names = ['France']

patches, sizes = patches.patches_from_region_list(region_list)

francais = True

data_patches = gather_data(patches, SOS_medecins=False, include_early=True)


data_patches[0] = data_patches[0].drop(['Hospital admissions', 'ICU admissions'], axis = 1)
data_patches[0].columns = ['Décès hospitaliers dus au Covid-19']

fit_total = lockdown.FitPatches(data_patches, names, sizes)

## sans second confinement
fit_total.dates_of_change = ['2020-03-16','2020-06-02', '2020-07-10']
fit_total.start_fit_init = '2020-03-01'
fit_total.dates_end_fit = ['2020-06-15', '2020-07-21', '2020-10-30']
if francais:
    fit_total.names_fit = ['confinement (jusqu\'au 2 juin)', 'après le 2 juin', 'deuxieme vague']
else:
    fit_total.names_fit = ['1st lockdown (until June 2nd)', 'after June 2nd', 'second wave']
fit_total.delays = np.array([[28], [15], [35]])
fit_total.fit_columns = [None, None, None] # None means all columns

# ## avec second confinement
# fit_total.dates_of_change = ['2020-03-16', '2020-06-02', '2020-07-10', '2020-10-30']
# fit_total.start_fit_init = '2020-03-19'
# fit_total.dates_end_fit = ['2020-06-15', '2020-07-21', '2020-10-30', '2020-12-01']
# if francais:
#     fit_total.names_fit = ['confinement (jusqu\'au 2 juin)', 'après le 2 juin', 'deuxieme vague', 'deuxieme confinement']
# else:
#     fit_total.names_fit = ['1st lockdown (until June 2nd)', 'after June 2nd', 'second wave', '2nd lockdown']
# fit_total.delays = np.array([[28], [15], [28], [10]])
# fit_total.fit_columns = [None, None, None, None] # None means all columns

fit_total.fit_patches(reference_column = 'Décès hospitaliers dus au Covid-19')

fig = fit_total.plot_fit_lockdown(francais = francais, display_main_legend = False)
fit_total.axs[0].set_title('Décès hospitaliers dus au Covid-19 en France')


