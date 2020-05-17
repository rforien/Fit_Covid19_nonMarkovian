#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:53:22 2020

@author: raphael
"""

import pandas as pd
import fit_lockdown as lockdown
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

data = pd.read_csv('donnees-hospitalieres-covid19-2020-05-13-19h00.csv', delimiter = ';')

deaths_early = pd.read_csv('deces_france_0101-1404.csv', index_col = 'jour')

# forget sex
data = data[data['sexe'] == 0]
# remove unused columns
deaths = data.pivot(index = 'jour', columns = 'dep', values = 'dc')

overseas = ['971', '972', '973', '974', '976']
# remove overseas
deaths = deaths.drop(overseas, axis=1)
#admissions = admissions.drop(overseas, axis = 1)
#reanimations = reanimations.drop(overseas, axis = 1)

IDF = ['92', '75', '77', '78', '91', '93', '94', '95']
GrandEst = ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88']
HautsdeFrance = ['02', '59', '60', '62', '80']
PACA = ['13', '84', '83', '04', '05', '06']
RhoneAlpes = ['01', '03', '07', '15', '26', '38', '42', '43', '63', '69', '73', '74']
Occitanie = ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82']
Bretagne = ['22', '29', '35', '56']
Loire = ['44', '49', '53', '72', '85']
Normandie = ['14', '27', '50', '61', '76']
Centre = ['18', '28', '36', '37', '41', '45']
Aquitaine = ['16', '17', '19', '23', '24', '33', '40', '47', '64', '79', '86', '87']
Bourgogne = ['21', '25', '39', '58', '70', '71', '89', '90']
Corse = ['2A', '2B']

green_regions = PACA + RhoneAlpes + Occitanie + Bretagne + Loire + Normandie + Centre + Aquitaine + Bourgogne


#regions = [IDF, GrandEst, HautsdeFrance, RhoneAlpes, PACA, Occitanie, 
#           Loire, Centre, Aquitaine, Bourgogne, Corse]
#names = ['Ile de France', 'Grand Est', 'Hauts de France', 'Auvergne Rhone Alpes', 
#         "Provence Alpes Cote d'Azur", 'Occitanie', 'Loire Atlantique',  
#         'Centre Val de Loire', 'Nouvelle Aquitaine', 'Bourgogne Franche Comté', 'Corse']
regions = [IDF, GrandEst + HautsdeFrance, green_regions]
names = ['Ile de France', 'Grand Est and Hauts-de-France', 'Green areas excluding Corsica']
colors = [cm.tab20(x) for x in np.tile(np.linspace(0,1,10,endpoint=False), 2)]
colors_fits = [cm.tab20(x + .05) for x in np.tile(np.linspace(0,1,10,endpoint=False),2)]

deaths_France = deaths.sum(axis=1)
deaths_France = pd.DataFrame(deaths_France, columns = ['deces'])
deaths_France = pd.concat((deaths_early['2020-02-15':'2020-03-17'], deaths_France), axis = 0)

fit_France = lockdown.Fitter(deaths_France, '2020-03-17', 25)
fit_France.fit_init('2020-03-01', '2020-03-24')

plt.figure(dpi = 200)
axes = plt.axes()
axes.set_ylabel('Cumulative number of deaths')
axes.set_title('Growth rate during the early phase of the epidemic')

axes.plot(fit_France.data['cumul'], label = 'Mainland France (r = %.2f)' % fit_France.r, color = colors[0])
axes.set_xticks(fit_France.data.index[0::9])
axes.set_xticklabels(fit_France.data.index[0::9])
axes.set_yscale('log')
n = np.size(fit_France.index_init)
y = np.exp(fit_France.regression_init.intercept + fit_France.r*np.arange(n))
axes.plot(fit_France.index_init, y, linestyle = 'dashdot', color = colors_fits[0])

for (i, r) in enumerate(regions):
    deaths_r = deaths[r].sum(axis = 1)
    fit = lockdown.Fitter(deaths_r, '2020-03-18', 23)
    fit.fit_init('2020-03-19', '2020-03-26')
    axes.plot(fit.data['cumul'], label = names[i] + ' (r = %.2f)' % fit.r, color = colors[i+1])
    n = np.size(fit.index_init)
    y = np.exp(fit.regression_init.intercept + fit.r*np.arange(n))
    axes.plot(fit.index_init, y, linestyle = 'dashdot', color = colors_fits[i+1])
    
axes.legend(loc = 'best')

