#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:43:43 2020

@author: raphael
"""

import pandas as pd
import fit_lockdown as lockdown
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('donnees-hospitalieres-nouveaux-covid19-2020-05-03-19h00.csv', delimiter = ';')

#deaths_early = pd.read_csv('deces_france_0101-1404.csv', index_col = 'jour')

# remove unused columns
admissions = data.pivot(index = 'jour', columns = 'dep', values = 'incid_hosp')
reanimations = data.pivot(index = 'jour', columns = 'dep', values = 'incid_rea')

plt.plot(admissions.sum(axis = 1), label = 'hospitalisations')
plt.plot(reanimations.sum(axis = 1), label = 'reanimations')
plt.yscale('log')
plt.legend(loc='best')

# remove overseas
admissions = admissions.drop(['971', '972', '973', '974', '976'], axis=1)

# Ile de France
adm_Idf = admissions[['92', '75', '77', '78', '91', '93', '94', '95']].sum(axis=1)
# switch to cumulative data
adm_Idf = np.cumsum(adm_Idf)

fit_idf_adm = lockdown.Fitter(adm_Idf, '2020-03-19', 12)

fit_idf_adm.fit_lockdown('2020-05-03')

fit_idf_adm.plot_fit()
fit_idf_adm.axes.set_title('Hospitalisations en Ile de France')

print(fit_idf.rE)