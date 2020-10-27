#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:27:20 2020

@author: raphael
"""

import numpy as np

from gather_data import gather_data
from build_patches import *
import fit_lockdown as lockdown

# region_list = [['Île de France'], ['Grand Est', 'Hauts-de-France'], ['Auvergne-Rhône-Alpe',
#               'Bourgogne-Franche-Comté', 'Bretagne', 'Centre-Val de Loire', 'Normandie',
#               'Nouvelle Aquitaine', 'Occitanie', 'Pays de la Loire', "Provence-Alpes-Côte d’Azur"]]
# names = ['Ile de France', 'Grand Est and Hauts-de-France', 'Rest of France']

region_list = [['Île de France'], ['Grand Est', 'Hauts-de-France', 'Bourgogne-Franche-Comté'], 
                  ['Auvergne-Rhône-Alpe', "Provence-Alpes-Côte d’Azur"],
                  ['Nouvelle Aquitaine', 'Occitanie'],
                  ['Bretagne', 'Centre-Val de Loire', 'Normandie', 'Pays de la Loire']]
names = ['Ile de France', 'Nord Est', 'Sud Est', 'Sud Ouest', 'Nord Ouest']

# region_list = [['Île de France', 'Grand Est', 'Hauts-de-France', 'Bourgogne-Franche-Comté', 
#                 'Auvergne-Rhône-Alpe', "Provence-Alpes-Côte d’Azur",
#                 'Nouvelle Aquitaine', 'Occitanie',
#                 'Bretagne', 'Centre-Val de Loire', 'Normandie', 'Pays de la Loire']]
# names = ['France']

# region_list = [['Île de France'], ["Provence-Alpes-Côte d’Azur"]]
# names = ['Ile de France', "Provence-Alpes-Cote d'Azur"]

patches, sizes = patches_from_region_list(region_list)

# patches, sizes = split_region("Provence-Alpes-Côte d’Azur")
# names = [patch[0] for patch in patches]


# patchIDF, sizeIDF = patches_from_regions(['Île de France'])
# names = ['Île de France', 'Bouches-du-Rhône', 'Rhône', 'Gironde', 'Bas-Rhin', 'Nord', 'Haute-Garonne', 'Hérault', 'Loire-Atlantique']
# patch_, sizes_ = patches_from_departments(names[1:])
# patches = patchIDF + patch_
# sizes = np.concatenate((sizeIDF, sizes_))

data_patches = gather_data(patches, SOS_medecins=False)

fit_total = lockdown.FitPatches(data_patches, names, sizes)

fit_total.dates_of_change = ['2020-03-16', '2020-05-11', '2020-06-02', '2020-07-10']
fit_total.dates_end_fit = ['2020-05-11', '2020-06-15', '2020-07-21', '2020-10-01']
fit_total.names_fit = ['Lockdown', 'After 11 May', 'After 2 June', 'After 10 July']
fit_total.delays = np.array([[18, 28, 28], [10, 15, 15], [10, 15, 15], [15, 21, 21]])

fit_total.fit_patches()
# fit_total.prepare_sir(.8, .005)
# fit_total.plot_delays()

fit_total.plot_fit_lockdown()

fit_total.compute_sir(.8, .006, '2020-10-31', Markov = False, two_step_measures = False)
# fit_total.sir[0].plot()
fit_total.plot_events()
# fit_total.plot_events(daily = False, logscale = False)

#fit_total.plot_fit_init(France, .6, .005)
# fit_total.plot_markov_vs_nonmarkov(.8, .005, logscale = False)
# fit_total.plot_immunity([.002, .005, .01], .8, '2020-09-30')
#print(fit_total._fit_reported(np.array([.6, 14.8, .18, 4.7, .9])))
#fit_total.fit_mcmc(5e3, np.array([.8, 14, .2, 7, .5]))
#fit_total.compute_sir(.6, f, end_of_run = '2020-04-17', Markov = False)
#fit_total.sir[0].lockdown_constants(-.05, 20)
#fit_total.plot_deaths_hosp(logscale = False)
#[sir.plot() for sir in fit_total.sir]
#fit_total.plot_SIR_deaths_hosp(logscale = True)

# shade areas depending on period
# use first day of icu total data to offset daily cumul