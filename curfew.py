#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:30:52 2020

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt

import fit_regions

p_reported = .8
ifr = .005

fitters_cur = []
fitters_cur.append(fit_regions.fit_GrandEst(p_reported, ifr, curfew = False))
fitters_cur.append(fit_regions.fit_PACA(p_reported, ifr, curfew = False))

fig2, axes2 = plt.subplots(1,2, dpi = 200, figsize=(6,9))
for (i, fitter) in enumerate(fitters_cur):
    data_lines, sir_lines = fitter.plot_events(axs = axes2[i], nb_xticks = 4)
predic_labels = 3*['Model predictions without curfew']
fig2.legend(tuple(data_lines + sir_lines), tuple(fitters_cur[0].events) + tuple(predic_labels), 
            loc=(.09, .85), fontsize = 12, ncol = 2)
plt.tight_layout(True)