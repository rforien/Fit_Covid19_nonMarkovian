#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:06:35 2020

@author: rforien
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fit = np.loadtxt('mcmc_fit_1806', delimiter = ',')
traj = np.loadtxt('mcmc_traj_1806', delimiter = ',')

names = ['Proportion of reported individuals', 'Mean infection to hospital admission delay',
         'Offset ratio (hosp)', 'Mean admission to death delay', 'Offset ratio (death)']
scale = [5, 1, 7, 1, 5]

if fit[-1] == 0:
    k = np.argmax(fit == 0)
    fit = fit[:k]
    traj = traj[:k,:]

i = np.argmin(fit)
print(traj[i,:])

def plot_coord(k, l):
    plt.figure()
    for j in np.arange(np.size(fit)):
        s = np.minimum(fit[j]/(3*fit[i]), 1)
        plt.scatter(traj[j,k], traj[j,l], color = cm.jet(s))

burn = 500
low = np.zeros(np.size(traj, axis = 1))
up = np.zeros(np.size(low))
for k in np.arange(np.size(traj, axis = 1)):
    plt.figure()
    plt.hist(traj[burn:,k], bins = int(np.size(fit)**.5), normed = True)
    plt.plot(traj[burn:,k], scale[k]*fit[burn:])
    plt.title(names[k])
    low[k] = np.quantile(traj[burn:,k], .05)
    up[k] = np.quantile(traj[burn:,k], .95)
    print('Confidence interval for ' + names[k] + ': [%.2f, %.2f]' % (low[k], up[k]))


'''
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
'''
