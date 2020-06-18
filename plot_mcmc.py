#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:06:35 2020

@author: rforien
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fit = np.loadtxt('mcmc_fit_good', delimiter = ',')
traj = np.loadtxt('mcmc_traj_good', delimiter = ',')

if fit[-1] == 0:
    k = np.argmax(fit == 0)
    fit = fit[:k]
    traj = traj[:k,:]

i = np.argmin(fit)

def plot_coord(k, l):
    plt.figure()
    for j in np.arange(np.size(fit)):
        s = np.minimum(fit[j]/(3*fit[i]), 1)
        plt.scatter(traj[j,k], traj[j,l], color = cm.jet(s))

plot_coord(1, 3)
plt.ylabel('Mean hosp to death delay')
plt.xlabel('Mean infection to hosp delay')
plt.show()

plot_coord(1, 2)
plt.ylabel('offset ratio')
plt.xlabel('Mean infection to hosp delay')
plt.show()

plot_coord(3, 4)
plt.xlabel('Mean hosp to death delay')
plt.ylabel('offset ratio')
plt.show()

plot_coord(0, 1)
plt.xlabel('Proportion of reported')
plt.ylabel('Mean infection to hosp delay')
plt.show()

plot_coord(0, 3)
plt.xlabel('Proportion of reported')
plt.ylabel('Mean hosp to death delay')
plt.show()

