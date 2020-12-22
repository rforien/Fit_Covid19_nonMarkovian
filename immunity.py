#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 22:50:11 2020

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt

import fit_regions

p_reported = 0.8

ifrs = [.003, .005, .01]

fitters_im = []
for f in ifrs:
    fitters_im.append(fit_regions.fit_IDF(p_reported, f))

fig = plt.figure(dpi=200)
axs = plt.axes()

for fitter in fitters_im:
    axs.plot(fitter.sir.times-fitter.sir.lockdown_time, 1-fitter.sir.traj[:,0], 
             label = 'f = %.2f%%' % (100*fitter.probas['Hospital deaths']))
    
axs.legend(loc='best', title='Infection fatality ratio')
tick_interval = int(np.size(fitters[0].sir.times)/6)
tick_times = (fitters_im[0].sir.times-fitters_im[0].sir.lockdown_time)[0::tick_interval]
labels = fitters_im[0].time_to_date(tick_times)
axs.set_xticks(tick_times)
axs.set_xticklabels(labels)
axs.grid(True)
axs.set_title('Predicted level of immunity in Île-de-France')
axs.set_ylabel('Proportion of infected individuals')
plt.tight_layout()