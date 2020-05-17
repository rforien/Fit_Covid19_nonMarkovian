#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:06:28 2020

@author: raphael
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fit_lockdown as lockdown

class FitPatches(object):
    lockdown_date = '2020-03-16'
    end_lockdown_fit = '2020-05-13'
    delay_deaths = 21
    delay_hosp = 13
    
    def __init__(self, deaths, hospitalisations, sizes):
        assert isinstance(deaths, pd.DataFrame) and isinstance(hospitalisations, pd.DataFrame)
        assert np.min(sizes) > 0
        self.n = np.size(sizes)
        self.sizes = sizes
        assert np.size(deaths.columns) == self.n and (hospitalisations.columns == deaths.columns).all()
        self.deaths = deaths
        self.hosp = hospitalisations
        self.names = deaths.columns
    
    def fit_patches(self):
        self.death_fitters = []
        self.hosp_fitters = []
        self.r = np.zeros(self.n)
        self.rE = np.zeros(self.n)
        self.deaths_at_lockdown = np.zeros(self.n)
        for (i, n) in enumerate(self.names):
            self.death_fitters.append(lockdown.Fitter(self.deaths[n], self.lockdown_date, self.delay_deaths))
            self.death_fitters[i].fit_init('2020-03-19', '2020-03-26')
            self.hosp_fitters.append(lockdown.Fitter(self.hosp[n], self.lockdown_date, self.delay_hosp))
            self.hosp_fitters[i].fit_lockdown(self.end_lockdown_fit)
            self.r[i] = self.death_fitters[i].r
            self.rE[i] = self.hosp_fitters[i].rE
            self.deaths_at_lockdown[i] = self.death_fitters[i].deaths_at_lockdown()
        print('Growth rates prior to lockdown: ', self.r)
        print('Growth rates during lockdown: ', self.rE)
        print('Deaths at lockdown: ', self.deaths_at_lockdown)
    
    def compute_sir(self, f, EI_dist, delay_death, delay_hosp, Markov = False):
        self.sir = []
        if Markov:
            for i in np.arange(self.n):
                self.sir.append(lockdown.SEIR_lockdown_mixed_delays(self.sizes[i], self.r[i], self.rE[i], f))
        else:
            for i in np.arange(self.n):
                self.sir.append(lockdown.SEIR_lockdown_mixed_delays(self.sizes[i], self.r[i], self.rE[i], f, EI_dist, delay_death))


