#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:11:48 2020

@author: raphael
"""

import pandas as pd
import fit_lockdown as lockdown
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

data_new = pd.read_csv('donnees-hospitalieres-nouveaux-covid19-2020-05-10-19h00.csv', delimiter = ';')

# remove unused columns
admissions = data_new.pivot(index = 'jour', columns = 'dep', values = 'incid_hosp')
reanimations = data_new.pivot(index = 'jour', columns = 'dep', values = 'incid_rea')

data = pd.read_csv('donnees-hospitalieres-covid19-2020-05-10-19h00.csv', delimiter = ';')

deaths_early = pd.read_csv('deces_france_0101-1404.csv', index_col = 'jour')

# forget sex
data = data[data['sexe'] == 0]
# remove unused columns
deaths = data.pivot(index = 'jour', columns = 'dep', values = 'dc')

overseas = ['971', '972', '973', '974', '976']
# remove overseas
deaths = deaths.drop(overseas, axis=1)
admissions = np.cumsum(admissions.drop(overseas, axis = 1))
reanimations = np.cumsum(reanimations.drop(overseas, axis = 1))

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


regions = [IDF, GrandEst, HautsdeFrance, RhoneAlpes, PACA, Occitanie, 
           Loire, Centre, Aquitaine, Bourgogne, Bretagne, Normandie, Corse]
names = ['Ile de France', 'Grand Est', 'Hauts de France', 'Auvergne Rhone Alpes', 
         "Provence Alpes Cote d'Azur", 'Occitanie', 'Loire Atlantique',  
         'Centre Val de Loire', 'Nouvelle Aquitaine', 'Bourgogne Franche Comté', 'Bretagne', 'Normandie', 'Corse']
colors = [cm.tab20(x) for x in np.tile(np.linspace(0,1,10,endpoint=False), 2)]
colors_fits = [cm.tab20(x + .05) for x in np.tile(np.linspace(0,1,10,endpoint=False),2)]

deaths_France = deaths.sum(axis=1)
deaths_France = pd.DataFrame(deaths_France, columns = ['deces'])
#deaths_France = pd.concat((deaths_early['2020-02-15':'2020-03-17'], deaths_France), axis = 0)

#ad_France = admissions.sum(axis = 1)
#rea_France = reanimations.sum(axis = 1)
#
#fit_d_France = lockdown.Fitter(deaths_France, '2020-03-18', 21)
#fit_d_France.fit_lockdown('2020-05-10')
#
#fit_ad_france = lockdown.Fitter(ad_France, '2020-03-19', 13)
#fit_ad_france.fit_lockdown('2020-05-10')
#
#fit_rea_france = lockdown.Fitter(rea_France, '2020-03-19', 15)
#fit_rea_france.fit_lockdown('2020-05-10')
#
#plt.figure(dpi=200, figsize = (10,4))
#axes = plt.axes()
#axes.set_yscale('log')
#
#axes.plot(fit_d_France.data['daily'], label = 'new deaths ($r_E$ = %.3f)' % fit_d_France.rE)
#axes.set_xticks(fit_d_France.data.index[0::7])
#axes.set_xticklabels(fit_d_France.data.index[0::7])
#
#n = np.size(fit_d_France.index_lockdown)
#y = np.exp(fit_d_France.regression_lockdown.intercept + fit_d_France.rE*np.arange(n))
#axes.plot(fit_d_France.index_lockdown, y, linestyle = 'dashdot')
#
#axes.plot(fit_ad_france.data['daily'], label = 'new hospital admissions ($r_E$ = %.3f)' % fit_ad_france.rE)
#n = np.size(fit_ad_france.index_lockdown)
#y = np.exp(fit_ad_france.regression_lockdown.intercept + fit_ad_france.rE*np.arange(n))
#axes.plot(fit_ad_france.index_lockdown, y, linestyle = 'dashdot')
#
#axes.plot(fit_rea_france.data['daily'], label = 'new ICU admissions ($r_E$ = %.3f)' % fit_rea_france.rE)
#n = np.size(fit_rea_france.index_lockdown)
#y = np.exp(fit_rea_france.regression_lockdown.intercept + fit_rea_france.rE*np.arange(n))
#plt.plot(fit_rea_france.index_lockdown, y, linestyle = 'dashdot')
#
#axes.legend(loc='best')
#axes.set_title('Decrease in hospital deaths, hospital admissions and\nICU admissions in mainland France under lockdown')

data_names = ['hospital deaths', 'hospital admissions', 'ICU admissions']

def fit_lockdown(region, name):
    fit_deaths = lockdown.Fitter(deaths[region].sum(axis=1), '2020-03-18', 21)
    fit_adm = lockdown.Fitter(admissions[region].sum(axis=1), '2020-03-19', 13)
    fit_rea = lockdown.Fitter(reanimations[region].sum(axis=1), '2020-03-19', 15)
    
    plt.figure(dpi = 200, figsize = (10,4))
    axes = plt.axes()
    axes.set_yscale('log')
    
    for (i, fit) in enumerate([fit_deaths, fit_adm, fit_rea]):
        fit.fit_lockdown('2020-05-10')
        n = np.size(fit.index_lockdown)
        y = np.exp(fit.regression_lockdown.intercept + fit.rE*np.arange(n))
        axes.plot(fit.data['daily'], label = 'new ' + data_names[i] + ' ($r_E$ = %.3f)' % fit.rE)
        axes.plot(fit.index_lockdown, y, linestyle = 'dashdot')
    
    axes.set_xticks(fit.data.index[0::7])
    axes.set_xticklabels(fit.data.index[0::7])
    axes.legend(loc='best')
    axes.set_title('Decrease in hospital deaths, hospital admissions and\nICU admissions in ' + name + ' under lockdown')

for (i, r) in enumerate(regions):
    fit_lockdown(r, names[i])