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

region_list = [['Île de France'], ['Grand Est', 'Hauts-de-France'], ['Auvergne-Rhône-Alpe',
              'Bourgogne-Franche-Comté', 'Bretagne', 'Centre-Val de Loire', 'Normandie',
              'Nouvelle Aquitaine', 'Occitanie', 'Pays de la Loire', "Provence-Alpes-Côte d’Azur"]]
names = ['Ile de France', 'Grand Est and Hauts-de-France', 'Rest of France']

# region_list = [['Île de France'], ['Grand Est', 'Hauts-de-France', 'Bourgogne-Franche-Comté'], 
#                 ['Auvergne-Rhône-Alpe', "Provence-Alpes-Côte d’Azur"],
#                 ['Nouvelle Aquitaine', 'Occitanie'],
#                 ['Bretagne', 'Centre-Val de Loire', 'Normandie', 'Pays de la Loire']]
# names = ['Ile de France', 'Nord Est', 'Sud Est', 'Sud Ouest', 'Nord Ouest']

# region_list = [['Île de France', 'Grand Est', 'Hauts-de-France', 'Bourgogne-Franche-Comté', 
#                 'Auvergne-Rhône-Alpe', "Provence-Alpes-Côte d’Azur",
#                 'Nouvelle Aquitaine', 'Occitanie',
#                 'Bretagne', 'Centre-Val de Loire', 'Normandie', 'Pays de la Loire']]
# names = ['France']

patches, sizes = patches_from_region_list(region_list)
data_patches = gather_data(patches, SOS_medecins=False)

fit_total = lockdown.FitPatches(data_patches, names, sizes)
fit_total.fit_patches()
fit_total.prepare_sir(.8, .005)

#fit_France = lockdown.MultiFitter(data_France)
#fit_France.fit(fit_total.lockdown_date, fit_total.lockdown_end_date,
#               fit_total.delays_lockdown, 'Lockdown')
#fit_France.fit(fit_total.lockdown_end_date, fit_total.end_post_lockdown,
#               fit_total.delays_post, 'After lockdown')
#fit_France.fit('2020-06-02', '2020-06-24', fit_total.delays_post, 'After 2 June')
#deaths_fit_France = lockdown.Fitter(France, fit_total.lockdown_date, 1)
#deaths_fit_France.fit_init('2020-03-01', fit_total.end_fit_init)
#print('Growth rates in France: ', deaths_fit_France.r, fit_France.params['Lockdown'][6], 
#      fit_France.params['After lockdown'][6], fit_France.params['After 2 June'][6])

# fit_total.rE[-1,:] = [.02, .02, .02]
fit_total.compute_sir(.8, .005, '2020-08-31', Markov = False, two_step_measures = False)
#fit_total.plot_fit_init(France, .6, .005)
# fit_total.plot_fit_lockdown()
# fit_total.plot_markov_vs_nonmarkov(.8, .005, logscale = False)
# fit_total.plot_immunity([.002, .005, .01], .8, '2020-07-31')
#print(fit_total._fit_reported(np.array([.6, 14.8, .18, 4.7, .9])))
#fit_total.fit_mcmc(5e3, np.array([.8, 14, .2, 7, .5]))
#fit_total.compute_sir(.6, f, end_of_run = '2020-04-17', Markov = False)
#fit_total.sir[0].lockdown_constants(-.05, 20)
#fit_total.plot_deaths_hosp(logscale = False)
#[sir.plot() for sir in fit_total.sir]
#fit_total.plot_SIR_deaths_hosp(logscale = True)
fit_total.plot_events()
fit_total.plot_events(daily = False, logscale = False)

# shade areas depending on period
# use first day of icu total data to offset daily cumul