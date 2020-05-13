#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:46:42 2020

@author: raphael
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

from . import sir

class Fitter(object):
    def __init__(self, cumul, lockdown_date, delay):
        assert lockdown_date in cumul.index
        assert delay >= 0 and type(delay)==int
        if isinstance(cumul, pd.DataFrame):
            self.data = cumul
            self.data.columns = ['cumul']
        elif isinstance(cumul, pd.Series):
            self.data = pd.DataFrame(cumul, columns = ['cumul'])
        else:
            assert False, 'cumul should be either a Series or a DataFrame.'
        self.lockdown_index = int(np.argwhere(self.data.index == lockdown_date))
        self.delay = delay
        self._compute_daily()
        
    def _compute_daily(self):
        self.data['daily'] = np.concatenate(([0], np.diff(self.data['cumul'].values)))
    
    def fit_init(self, start, end):
        assert start in self.data.index
        assert end in self.data.index
        y = np.log(self.data[start:end]['cumul'].values)
        x = np.arange(np.size(y))
        self.regression_init = stats.linregress(x, y)
        self.r = self.regression_init.slope
        self.index_init = self.data[start:end].index
    
    def fit_lockdown(self, end):
        assert end in self.data.index
        start = self.data.index[self.lockdown_index + self.delay]
        y = np.log(self.data[start:end]['daily'].values)
        x = np.arange(np.size(y))
        self.regression_lockdown = stats.linregress(x, y)
        self.rE = self.regression_lockdown.slope
        self.index_lockdown = self.data[start:end].index
    
#    def compute_Re(self):
#        assert hasattr(self, 'regression_init') and hasattr(self, 'regression_lockdown')
#        self.RE1 = 1 + (self.rE/self.r)*(self.R0 - 1)
#        self.RE2 = self.R0**(self.rE/self.r)
#    
#    def plot_Re(self):
#        assert hasattr(self, 'regression_init') and hasattr(self, 'regression_lockdown')
#        r_values = np.linspace(self.R0[0], self.R0[1], 100)
#        RE1 = 1 + (self.rE/self.r)*(r_values-1)
#        RE2 = r_values**(self.rE/self.r)
#        plt.figure(dpi = 200)
#        self.ax_RE = plt.axes()
#        self.ax_RE.plot(r_values, RE1, label = '$1+(r_E/r)(R_0-1)$')
#        self.ax_RE.plot(r_values, RE2, label = '$R_0^{r_E/r}$')
#        self.ax_RE.set_xlabel('$R_0$')
#        self.ax_RE.set_title('Possible values of $R_E$ during lockdown')
#        self.ax_RE.legend(loc='best')
    
    def plot_fit(self):
        plt.figure(dpi = 200, figsize = (10,5))
        self.axes = plt.axes()
        self.axes.plot(self.data['cumul'], label = 'cumulative')
        self.axes.plot(self.data['daily'], label = 'daily')
        if hasattr(self, 'regression_init'):
            n_init = np.size(self.index_init)
            y_init = np.exp(self.regression_init.intercept + self.r*np.arange(n_init))
            self.axes.plot(self.index_init, y_init, label = '$r$ = %.3f' % self.r, linestyle = 'dashdot')
        if hasattr(self, 'regression_lockdown'):
            n_lockdown = np.size(self.index_lockdown)
            y_lockdown = np.exp(self.regression_lockdown.intercept + self.rE*np.arange(n_lockdown))
            self.axes.plot(self.index_lockdown, y_lockdown, label = '$r_E$ = %.3f' % self.rE, linestyle = 'dashdot')
        self.axes.set_yscale('log')
        self.axes.legend(loc='best')
        self.axes.set_xticks(self.data.index[0::7])
        self.axes.set_xticklabels(self.data.index[0::7])
        self.axes.tick_params(axis='x', labelsize=9)
    
    def deaths_at_lockdown(self):
        assert hasattr(self, 'regression_init')
        t = self.lockdown_index - int(np.argwhere(self.data.index == self.index_init[0]))
        return np.exp(self.regression_init.intercept + self.r*t)
        
        
