#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:30:42 2020

@author: raphael
"""

import matplotlib.pyplot as plt
import numpy as np

import fit_lockdown as lockdown

N = 20e6
r = .3
rE = -.06
f = .005
deaths_at_lockdown = 50

lockdown_length = 55

delay = 15
delay_dist = np.array([[delay, 1]])

EI_dist = np.array([[2, 7, .8], [2, 11, .2]])
I_dist = EI_dist[:,1:]
incubation_time = np.sum(EI_dist[:,0]*EI_dist[:,2])
generation_time = np.sum(EI_dist[:,1]*EI_dist[:,2])

'''
sir_markov = lockdown.SIR_lockdown_mixed_delays(N, r, rE, f, generation_time, delay_dist)
seir_markov = lockdown.SEIR_lockdown_mixed_delays(N, r, rE, f, generation_time, incubation_time, delay_dist)
sir_nonmarkov = lockdown.SIR_nonMarkov(N, r, rE, f, I_dist, delay_dist)
seir_nonmarkov = lockdown.SEIR_nonMarkov(N, r, rE, f, EI_dist, delay_dist)

models = [sir_markov, seir_markov, sir_nonmarkov, seir_nonmarkov]
names = ['Markovian SIR', 'Markovian SEIR', 'non-Markovian SIR', 'non-Markovian SEIR']
lstyles = ['solid', 'dashed', 'dashdot', (0, (3, 1, 1, 1, 1, 1))]


for (i,m) in enumerate(models):
    print(names[i])
    m.calibrate(deaths_at_lockdown)
    m.run_full(lockdown_length, 0, 1)
    m.compute_deaths()
'''

fig, axs = plt.subplots(1,2, dpi=200, figsize = (13,4.5))

for (i,m) in enumerate(models):
    axs[0].plot(m.times_death, m.deaths, label = names[i], linestyle = lstyles[i])
    axs[1].plot(1 + np.arange(np.size(m.daily_deaths)), m.daily_deaths, label = names[i], linestyle = lstyles[i])

start = 50
end = 100

axs[0].legend(loc='best')
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[0].set_title('Cumulative number of deaths')
axs[1].set_title('Daily number of deaths')
axs[0].set_xlabel('time (days)')
axs[1].set_xlabel('time (days)')
axs[0].set_xlim((start, end))
axs[1].set_xlim((start, end))

#axs[1].plot(sir_markov.times_death, 1e3*np.exp(rE*sir_markov.times_death))
axs[0].set_ylim((2.5e3, 1.5e4))
axs[1].set_ylim((1e1, 2e3))