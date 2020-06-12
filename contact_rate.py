#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:46:21 2020

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
import fit_lockdown as lockdown

rho = np.linspace(-.6, 1, 200)

E_dist = [3,0]+[2,1]*lockdown.beta_dist(2, 2, 30)
I_dist = np.concatenate(([1, .8]*([3, 0] + [2, 1]*lockdown.beta_dist(2, 2, 20)),
                  [1, .2]*([5, 0] + [10, 1]*lockdown.beta_dist(2, 2, 20))), axis = 0)
EI_covid = lockdown.product_dist(E_dist, I_dist)

incub = np.sum(E_dist[:,0]*E_dist[:,1])
infect = np.sum(I_dist[:,0]*I_dist[:,1])

# E_dist = lockdown.beta_dist(1.5, 1.5, 30)
# E_dist[:,0] = 2.5+3*E_dist[:,0]
# I_dist = lockdown.beta_dist(1.5, 1.2, 30)
# I_dist[:,0] = 2+(5*(1.5+1.2)/1.5)*I_dist[:,0]

#E_dist = np.array([[3, .6], [4, .2], [5, .2]])
#I_dist = np.array([[3, .2], [5, .4], [10, .4]])

# EI_beta = lockdown.product_dist(E_dist, I_dist)

l = 2
E_dist = lockdown.gamma_dist(incub, 1, 300)
I_dist = lockdown.gamma_dist(infect/l, l, 300)

EI_gamma = lockdown.product_dist(E_dist, I_dist)


EI_fix = np.array([[incub, infect, 1]])

# EI_bimod = np.array([[4, 2, .5], [4, 14, .5]])

# sir_beta = lockdown.SEIR_nonMarkov(10, 1, 1, .1, EI_beta, np.array([[1, 1]]))
sir_gamma = lockdown.SEIR_nonMarkov(10, 1, 1, .1, EI_gamma, np.array([[1, 1]]))
sir_markov = lockdown.SEIR_lockdown(10, 1, 1, .1, infect, 1, incub)
sir_fix = lockdown.SEIR_nonMarkov(10, 1, 1, .1, EI_fix, np.array([[1, 1]]))
sir_covid = lockdown.SEIR_nonMarkov(10, 1, 1, .1, EI_covid, np.array([[1, 1]]))
# sir_bimod = lockdown.SEIR_nonMarkov(10, 1, 1, .1, EI_bimod, np.array([[1, 1]]))

S = [sir_markov, sir_fix, sir_covid, sir_gamma]
names = ['Markovian SEIR model', 'SEIR model with fixed durations',
         'non-Markovian SEIR model (COVID-19 like)', 'non-Markovian SEIR model (Gamma)']

plt.figure(dpi=250)

contact_rates = np.zeros(np.size(rho))

for (i, sir) in enumerate(S):
    for (j, r) in enumerate(rho):
        if i==0 and r < np.max([-incub**-1, -infect**-1]):
            contact_rates[j] = np.nan
        else:
            contact_rates[j] = sir.contact_rate(r)
    plt.plot(rho, infect*contact_rates, label = names[i], linewidth = 1.2)

#plt.vlines((-.049, .27), 0, 1e2, linestyle = 'dashed', linewidth = 1)
    
plt.vlines((-.06, .3), 0, 7, linewidth = 1)
plt.hlines((0,7), -.06, .3, linewidth = 1)

plt.legend(loc = 'best')
plt.yscale('log')
plt.grid(True)
plt.ylabel(r'$R_0$')
plt.xlabel(r'growth rate ($\rho$)')
#plt.xlim((-.06, .3))
#plt.ylim((0, 7))