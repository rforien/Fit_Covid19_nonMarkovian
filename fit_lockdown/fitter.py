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
import scipy.optimize as optim
import datetime as date

class MultiFitter(object):
    date_format = '%Y-%m-%d'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fit_colors = ['#d62728', '#8c564b', '#00AA00']
    
    def __init__(self, cumul):
        if isinstance(cumul, pd.Series):
            self.cumul = pd.DataFrame(cumul)
        elif isinstance(cumul, pd.DataFrame):
            self.cumul = cumul
        else:
            assert False, 'cumul should be either a Series or a DataFrame.'
        self.n = np.size(cumul, axis = 1)
        self.columns = cumul.columns
        self.daily = self.cumul.diff()
        self.params = pd.DataFrame()
        self.scales = pd.DataFrame()
        self.starts = pd.DataFrame()
        self.end = pd.DataFrame()
    
    def fit(self, start, end, delays, fit_key):
        assert end in self.cumul.index
        assert np.size(delays) == self.n and np.min(delays) >= 0
        # compute starting dates, lengths and scales
        starts = []
        lengths = np.zeros(self.n)
        scale = np.zeros(2*self.n)
        start_date = date.datetime.strptime(start, self.date_format)
        end_date = date.datetime.strptime(end, self.date_format)
        for (i, col) in enumerate(self.columns):
            d = start_date + date.timedelta(days = int(delays[i]))
            starts.append(d.strftime(self.date_format))
            assert starts[i] in self.cumul.index
            lengths[i] = (end_date-d).days+1
            scale[i] = self.cumul[col][starts[i]]
            scale[self.n+i] = -(self.cumul[col][end]-
                 self.cumul[col][starts[i]])
        init_params = np.concatenate((np.ones(2*self.n), [-.1]))
        result = optim.minimize(self._error, init_params, args = (
                starts, end, lengths, scale), method = 'Nelder-Mead')
        self.params[fit_key] = result.x
        self.scales[fit_key] = scale
        self.starts[fit_key] = starts
        self.end[fit_key] = [end]
        
    def _error(self, params, starts, end, lengths, scale):
        E = 0
        for (i, col) in enumerate(self.columns):
            x = self.cumul[col][starts[i]:end].values
            y = self.exp_fit(params, i, scale, np.arange(lengths[i]))
            E += np.sum(np.abs(1-y/x)**2)
        return E
    
    def exp_fit(self, params, i, scale, t):
        return (scale[i]*params[i] + scale[self.n+i]*params[self.n+i]*(
                np.exp(params[2*self.n]*t)-1))
    
    def exp_fit_daily(self, params, i, scale, t):
        return (scale[self.n+i]*params[self.n+i]*
                params[2*self.n]*np.exp(params[2*self.n]*t))
    
    def fit_value_at(self, column, fit, at_date, daily = False):
        assert fit in self.params.columns
        assert column in self.columns
        i = np.argwhere(self.columns == column)[0][0]
        start = date.datetime.strptime(self.starts[fit][i], self.date_format)
        d = date.datetime.strptime(at_date, self.date_format)
        t = (d-start).days
        if daily:
            return self.exp_fit_daily(self.params[fit], i, self.scales[fit], t)
        else:
            return self.exp_fit(self.params[fit], i, self.scales[fit], t)
    
    def plot(self, axes = None, fits = None):
        tick_interval = 20
        if axes == None:
            plt.figure(dpi = 200)
            axes = plt.axes()
        if fits == None:
            fits = self.params.columns
        else:
            for fit in fits:
                assert fit in self.params.columns
        data_lines = []
        for (i, col) in enumerate(self.columns):
            line, = axes.plot(self.daily[col].index, self.daily[col].values, 
                      linestyle = 'dashed', color = self.colors[i])
            data_lines.append(line)
            for (j, fit) in enumerate(fits):
                index = self.cumul[self.starts[fit][i]:self.end[fit][0]].index
                line, = axes.plot(index, self.exp_fit_daily(self.params[fit].values, i,
                            self.scales[fit], np.arange(np.size(index))), color = self.fit_colors[j])
                if i == 0:
                    line.set_label(fit + r': $\rho$ = %.3f' % self.params[fit][2*self.n])
        axes.set_yscale('log')
        axes.legend(loc = 'best')
        axes.grid(True)
        axes.set_xticks(self.cumul.index[0::tick_interval])
        axes.set_xticklabels(self.cumul.index[0::tick_interval])
        self.axes = axes
        return data_lines

class Fitter(object):
    date_format = '%Y-%m-%d'
    
    def __init__(self, cumul, lockdown_date, delay):
        self.lockdown_date = date.datetime.strptime(lockdown_date, self.date_format)
        assert delay >= 0 and type(delay)==int
        if isinstance(cumul, pd.DataFrame):
            self.data = cumul
            self.data.columns = ['cumul']
        elif isinstance(cumul, pd.Series):
            self.data = pd.DataFrame(cumul)
            self.data.columns = ['cumul']
        else:
            assert False, 'cumul should be either a Series or a DataFrame.'
        self.n = np.size(self.data['cumul'].values)
#        self.lockdown_index = int(np.argwhere(self.data.index == lockdown_date))
        self.delay = date.timedelta(days = delay)
        self._compute_daily()
        
    def _compute_daily(self):
        self.data['daily'] = np.concatenate(([0], np.diff(self.data['cumul'].values)))
    
    def average_daily(self, d = 3):
        self.data['averaged'] = np.zeros(self.n)
        self.data['averaged'].values[d:-d] = (self.data['cumul'].values[2*d:] - self.data['cumul'].values[:-2*d])/(2*d)
        for i in np.arange(d):
            self.data['averaged'].values[i] = (self.data['cumul'].values[i+d] - self.data['cumul'].values[0])/(i+d)
            self.data['averaged'].values[self.n-i-1] = (self.data['cumul'].values[-1] - self.data['cumul'].values[-(d+i)])/(i+d)
    
    def fit_init(self, start, end):
        assert start in self.data.index
        assert end in self.data.index
        y = np.log(self.data[start:end]['cumul'].values)
        x = np.arange(np.size(y))
        self.regression_init = stats.linregress(x, y)
        self.r = self.regression_init.slope
        self.index_init = self.data[start:end].index
        self.n_init = np.size(self.index_init)
        self.date_init_fit = date.datetime.strptime(start, self.date_format)
    
#    def _fit_lockdown(self, end):
#        assert end in self.data.index
#        self.average_daily()
#        start = self.data.index[self.lockdown_index + self.delay]
#        y = np.log(self.data[start:end]['averaged'].values)
#        x = np.arange(np.size(y))
#        self.regression_lockdown = stats.linregress(x, y)
#        self.rE = self.regression_lockdown.slope
#        self.index_lockdown = self.data[start:end].index
    
    def fit_lockdown(self, end):
        assert end in self.data.index
        self.end_lock = end
        self.start_lock = (self.lockdown_date + self.delay).strftime(self.date_format)
        assert self.start_lock in self.data.index
        self.n_lock = np.size(self.data[self.start_lock:self.end_lock]['cumul'].values)
        self.N0_lock = self.data['cumul'][self.start_lock]
        self.scale_lock = (self.data['cumul'][self.end_lock] - self.data['cumul'][self.start_lock])/self.n_lock
        init_params = np.array([.1, -1])
        self.result_lockdown = optim.minimize(self._error, init_params, 
                                              args = (self.N0_lock, self.start_lock, self.end_lock, self.n_lock, self.scale_lock))
        self.fit_params = self.result_lockdown.x
        self.rE = self.fit_params[0]
        self.index_lockdown = self.data[self.start_lock:self.end_lock].index
    
    def _error(self, params, N0, start, end, n, scale = 1):
        y = N0 + scale*params[1]*(np.exp(params[0]*np.arange(n))-1)
        x = self.data[start:end]['cumul'].values
        # print(x)
        # print(y)
        E = np.sum(np.abs(1-y/x)**2)
        # print(params, E)
        return E
    
    def best_fit_lock_cumul(self):
        assert hasattr(self, 'fit_params')
        return self.N0_lock + self.scale_lock*self.fit_params[1]*(np.exp(self.fit_params[0]*np.arange(self.n_lock))-1)
    
    def best_fit_lock_daily(self):
        assert hasattr(self, 'fit_params')
        return self.scale_lock*self.fit_params[1]*self.fit_params[0]*np.exp(self.fit_params[0]*np.arange(self.n_lock))
    
    def best_fit_init_cumul(self):
        assert hasattr(self, 'regression_init')
        return np.exp(self.regression_init.intercept + self.r*np.arange(self.n_init))
    
    def best_fit_init_daily(self):
        assert hasattr(self, 'regression_init')
        return self.r*np.exp(self.regression_init.intercept + self.r*np.arange(self.n_init))
    
    def fit_post_lockdown(self, start, end):
        assert start in self.data.index and end in self.data.index
        self.n_post = np.size(self.data[start:end]['cumul'].values)
        self.N0_post = self.data['cumul'][start]
        # self.scale_post = (self.data['cumul'][end]-self.data['cumul'][start])/self.n_lock
        self.scale_post = 1
        init_params = np.array([-.1, -20])
        self.result_post_lockdown = optim.minimize(self._error, init_params,
                                                   args = (self.N0_post, start, end, self.n_post), method='Nelder-Mead')
        self.fit_params_post = self.result_post_lockdown.x
        self.r_post = self.fit_params_post[0]
        self.index_post_lockdown = self.data[start:end].index
    
    def best_fit_post_cumul(self):
        assert hasattr(self, 'fit_params_post')
        return self.N0_post + self.scale_post*self.fit_params_post[1]*(np.exp(self.fit_params_post[0]*np.arange(self.n_post))-1)
    
    def best_fit_post_daily(self):
        assert hasattr(self, 'fit_params_post')
        return self.scale_post*self.fit_params_post[1]*self.fit_params_post[0]*np.exp(self.fit_params_post[0]*np.arange(self.n_post))
    
    def plot_fit(self, tick_interval = 7):
        assert type(tick_interval) == int and tick_interval > 0
        plt.figure(dpi = 200, figsize = (10,5))
        self.axes = plt.axes()
        self.axes.plot(self.data['cumul'], label = 'cumulative')
        self.axes.plot(self.data['daily'], label = 'daily')
        if hasattr(self, 'regression_init'):
            p = self.axes.plot(self.index_init, self.best_fit_init_daily(), label = '$r$ = %.3f' % self.r, linestyle = 'dashdot')
            self.axes.plot(self.index_init, self.best_fit_init_cumul(), linestyle = 'dashdot', color = p[0].get_color())
#        if hasattr(self, 'regression_lockdown'):
#            n_lockdown = np.size(self.index_lockdown)
#            y_lockdown = np.exp(self.regression_lockdown.intercept + self.rE*np.arange(n_lockdown))
#            self.axes.plot(self.index_lockdown, y_lockdown, label = '$r_E$ = %.3f' % self.rE, linestyle = 'dashdot')
        if hasattr(self, 'result_lockdown'):
            p = self.axes.plot(self.index_lockdown, self.best_fit_lock_daily(), label = '$r_E$ = %.3f' % self.rE, linestyle = 'dashdot')
            self.axes.plot(self.index_lockdown, self.best_fit_lock_cumul(), linestyle = 'dashdot', color = p[0].get_color())
        self.axes.set_yscale('log')
        self.axes.legend(loc='best')
        self.axes.set_xticks(self.data.index[0::tick_interval])
        self.axes.set_xticklabels(self.data.index[0::tick_interval])
        self.axes.tick_params(axis='x', labelsize=9)
    
    def deaths_at_lockdown(self):
        assert hasattr(self, 'regression_init')
        t = (self.lockdown_date - self.date_init_fit).days
        return np.exp(self.regression_init.intercept + self.r*t)
        
        
