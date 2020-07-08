#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:27:20 2020

@author: raphael
"""

import pandas as pd
import numpy as np

import fit_lockdown as lockdown

data = pd.read_csv('donnees-hospitalieres-covid19-2020-07-07-19h00_corrected.csv', delimiter = ';')
data_new = pd.read_csv('donnees-hospitalieres-nouveaux-covid19-2020-07-07-19h00.csv', delimiter = ';')

deaths_early = pd.read_csv('deces_france_0101-1404.csv', index_col = 'jour')

# forget sex
data = data[data['sexe'] == 0]
# remove unused columns
deaths = data.pivot(index = 'jour', columns = 'dep', values = 'dc')

# remove dataoverseas
overseas = ['971', '972', '973', '974', '976']
deaths = deaths.drop(overseas, axis=1)

data['cumul_hosp'] = data['hosp'] + data['rad'] + data['dc']

admissions = data.pivot(index = 'jour', columns = 'dep', values = 'cumul_hosp')
admissions.drop(overseas, axis = 1)

rea = data_new.pivot(index = 'jour', columns = 'dep', values = 'incid_rea')

N_france = 67e6

IDF = ['92', '75', '77', '78', '91', '93', '94', '95']
N_idf = 12.21e6
GrandEst = ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88']
N_GE = 5.55e6
HautsdeFrance = ['02', '59', '60', '62', '80']
N_HdF = 6e6

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

Out = PACA + RhoneAlpes + Occitanie + Bretagne + Loire + Normandie + Centre + Aquitaine + Bourgogne

France = deaths[Out + IDF + GrandEst + HautsdeFrance].sum(axis=1)
France = pd.DataFrame(France, columns = ['deces'])
France = pd.concat((deaths_early['2020-02-15':'2020-03-17'], France), axis = 0)

N_out = N_france - N_idf - N_GE - N_HdF

deaths_IDF = deaths[IDF].sum(axis=1)
deaths_GEHdF = deaths[GrandEst + HautsdeFrance].sum(axis=1)
deaths_out = deaths[Out].sum(axis=1)
deaths_patches = pd.concat((deaths_IDF, deaths_GEHdF, deaths_out), axis = 1)
deaths_patches.columns = ['Ile de France', 'Grand Est and Hauts-de-France', 'Rest of France']

admis_IDF = admissions[IDF].sum(axis=1)
admis_GEHdF = admissions[GrandEst + HautsdeFrance].sum(axis=1)
admis_out = admissions[Out].sum(axis=1)
admis_patches = pd.concat((admis_IDF, admis_GEHdF, admis_out), axis = 1)
admis_patches.columns = deaths_patches.columns

admis_France = admissions[IDF + GrandEst + HautsdeFrance + Out].sum(axis = 1)
rea_France = rea[IDF + GrandEst + HautsdeFrance + Out].sum(axis = 1).cumsum()

rea_GEHdF = rea[GrandEst + HautsdeFrance].sum(axis = 1).cumsum()
rea_IdF = rea[IDF].sum(axis = 1).cumsum()
rea_out = rea[Out].sum(axis = 1).cumsum()

data_IDF = pd.concat((admis_IDF, deaths_IDF, rea_IdF), axis = 1)
data_IDF.columns = ['Hospital admissions', 'Hospital deaths', 'ICU admissions']

data_GEHdF = pd.concat((admis_GEHdF, deaths_GEHdF, rea_GEHdF), axis = 1)
data_GEHdF.columns = data_IDF.columns

data_out = pd.concat((admis_out, deaths_out, rea_out), axis = 1)
data_out.columns = data_IDF.columns

data_France = pd.concat((admis_France, France, rea_France), axis = 1)
data_France.columns = data_IDF.columns

data_patches = [data_IDF, data_GEHdF, data_out]
names = ['Ile de France', 'Grand Est and Hauts-de-France', 'Rest of France']

#E_dist = lockdown.beta_dist(2, 2)
#E_dist[:,0] = 2+3*E_dist[:,0]
#I_dist = lockdown.beta_dist(2, 1.1)
#I_dist[:,0] = 2+10*I_dist[:,0]
#EI_dist = lockdown.product_dist(E_dist, I_dist)
#EI_dist = np.array([[3.5, 2, .17], [3.5, 5, .67], [3.5, 14, .16]]) # best fit
EI_dist = np.array([[3.5, 3, .8], [3.5, 10, .2]])
f = .005
#f = .0037 # estimate from germany
#delays = np.array([[21, 1]])
#delay_death = np.transpose(np.vstack((np.linspace(11, 25, 20), np.ones(20)/20)))
#delay_hosp = np.transpose(np.vstack((np.linspace(6, 18, 10), .1*np.ones(10))))
delay_hosp = [6, 0] + [10, 1]*lockdown.beta_dist(1.5, 1.2, 20)
delay_death = ([7, 0] + [20, 1]*lockdown.beta_dist(2, 1.5, 20))

# MigMat = np.zeros((4, 3, 3))
# for i in np.arange(4):
#     MigMat[i] = .01*np.array([[-1, .5, .5], [.5, -1, .5], [.5, .5, -1]])



fit_total = lockdown.FitPatches(data_patches, names, [N_idf, N_GE + N_HdF, N_out])
fit_total.fit_patches()

fit_France = lockdown.MultiFitter(data_France)
fit_France.fit(fit_total.lockdown_date, fit_total.lockdown_end_date,
               fit_total.delays_lockdown, 'Lockdown')
fit_France.fit(fit_total.lockdown_end_date, fit_total.end_post_lockdown,
               fit_total.delays_post, 'After lockdown')
fit_France.fit('2020-06-02', '2020-06-24', fit_total.delays_post, 'After 2 June')
deaths_fit_France = lockdown.Fitter(France, fit_total.lockdown_date, 1)
deaths_fit_France.fit_init('2020-03-01', fit_total.end_fit_init)
print('Growth rates in France: ', deaths_fit_France.r, fit_France.params['Lockdown'][6], 
      fit_France.params['After lockdown'][6], fit_France.params['After 2 June'][6])

# fit_total.rE[-1,:] = [.02, .02, .02]
fit_total.compute_sir(.6, f, '2020-08-31', Markov = False)
#fit_total.plot_fit_init(France, .6, .005)
# fit_total.plot_fit_lockdown()
#fit_total.plot_markov_vs_nonmarkov(.6, .005, logscale = True)
# fit_total.plot_immunity([.002, .005, .01], .6)
#print(fit_total._fit_reported(np.array([.6, 14.8, .18, 4.7, .9])))
#fit_total.fit_mcmc(5e3, np.array([.8, 14, .2, 7, .5]))
#fit_total.compute_sir(.6, f, end_of_run = '2020-04-17', Markov = False)
#fit_total.sir[0].lockdown_constants(-.05, 20)
fit_total.plot_deaths_hosp(logscale = False)
#[sir.plot() for sir in fit_total.sir]
# fit_total.plot_SIR_deaths_hosp(logscale = False)

# shade areas depending on period
# try fsolve with different starting point when fit doesn't work