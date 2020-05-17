#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 11:27:20 2020

@author: raphael
"""

import pandas as pd
import numpy as np

import fit_lockdown as lockdown

data = pd.read_csv('donnees-hospitalieres-covid19-2020-05-13-19h00.csv', delimiter = ';')

deaths_early = pd.read_csv('deces_france_0101-1404.csv', index_col = 'jour')

# forget sex
data = data[data['sexe'] == 0]
# remove unused columns
deaths = data.pivot(index = 'jour', columns = 'dep', values = 'dc')

# remove overseas
overseas = ['971', '972', '973', '974', '976']
deaths = deaths.drop(overseas, axis=1)

data['cumul_hosp'] = data['hosp'] + data['rad'] + data['dc']

admissions = data.pivot(index = 'jour', columns = 'dep', values = 'cumul_hosp')
admissions.drop(overseas, axis = 1)

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

N_out = N_france - N_idf - N_GE - N_HdF

deaths_IDF = deaths[IDF].sum(axis=1)
deaths_GEHdF = deaths[GrandEst + HautsdeFrance].sum(axis=1)
deaths_out = deaths[Out].sum(axis=1)
deaths_patches = pd.concat((deaths_IDF, deaths_GEHdF, deaths_out), axis = 1)
deaths_patches.columns = ['Ile de France', 'Grand Est et Hauts-de-France', 'Reste de la France']

admis_IDF = admissions[IDF].sum(axis=1)
admis_GEHdF = admissions[GrandEst + HautsdeFrance].sum(axis=1)
admis_out = admissions[Out].sum(axis=1)
admis_patches = pd.concat((admis_IDF, admis_GEHdF, admis_out), axis = 1)
admis_patches.columns = deaths_patches.columns

fit_total = lockdown.FitPatches(deaths_patches, admis_patches, [N_idf, N_GE + N_HdF, N_out])
fit_total.fit_patches()