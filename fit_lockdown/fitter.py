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
import matplotlib.cm as cm
import itertools as itertl

class MultiFitter(object):
    date_format = '%Y-%m-%d'
    reference_date = '2020-01-01'
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
        index = [(col, i) for (col, i) in itertl.product(self.columns, [0, 1])] + [('growth rate', 0)]
        self.params = pd.DataFrame(index = pd.MultiIndex.from_tuples(index))
        self.fit_columns = pd.DataFrame(index = self.columns)
        self.scales = pd.DataFrame(index = pd.MultiIndex.from_product([self.columns, [0, 1]]))
        self.starts = pd.DataFrame(index = self.columns)
        self.end = pd.DataFrame()
        self.ref_datetime = date.datetime.strptime(self.reference_date, self.date_format)
    
    def fit(self, start, end, delays, fit_key, columns = None):
        assert end in self.cumul.index
        if columns == None:
            columns = self.columns
            self.fit_columns[fit_key] = self.n*[True]
        else:
            self.fit_columns[fit_key] = [col in columns for col in self.columns]
            assert self.fit_columns[fit_key].any(), "Invalid columns argument"
        m = np.sum(self.fit_columns[fit_key])
        assert np.size(delays) == m and np.min(delays) >= 0
        print('fitting ' + fit_key)
        # compute starting dates, lengths and scales
        self.starts[fit_key] = self.n*['']
        lengths = np.zeros(self.n)
        self.scales[fit_key] = np.zeros(2*self.n)
        start_date = date.datetime.strptime(start, self.date_format)
        end_date = date.datetime.strptime(end, self.date_format)
        for (i, col) in enumerate(self.columns):
            if not self.fit_columns[fit_key][col]:
                continue
            j = np.sum(self.fit_columns[fit_key].values[:i])
            d = start_date + date.timedelta(days = int(delays[j]))
            self.starts[fit_key][col] = d.strftime(self.date_format)
            assert self.starts[fit_key][col] in self.cumul.index
            lengths[i] = (end_date-d).days+1
            # self.scales[fit_key][col] = [self.cumul[col][self.starts[fit_key][col]], 
            #             (self.cumul[col][end]-self.cumul[col][self.starts[fit_key][col]])]
            self.scales[fit_key][col] = np.array([self.cumul[col][self.starts[fit_key][col]], 
                        self.daily[col][self.starts[fit_key][col]]])/10
        self.end[fit_key] = [end]
        init_params = np.concatenate((10*np.ones(2*m), [-.01]))
        # init_params = np.ones(2*self.n+1)
        result = optim.minimize(self._error, init_params, args = (
                fit_key, lengths), method = 'Nelder-Mead')
        self.params[fit_key] = (2*self.n+1)*[np.nan]
        for (i, col) in enumerate(self.columns):
            if not self.fit_columns[fit_key][col]:
                continue
            j = np.sum(self.fit_columns[fit_key].values[:i])
            self.params[fit_key][col] = [result.x[j], result.x[m+j]]
        print(result.x[-1])
        self.params[fit_key]['growth rate'] = result.x[-1]
        
    def _error(self, params, fit_key, lengths):
        E = 0
        for (i, col) in enumerate(self.columns):
            if not self.fit_columns[fit_key][col]:
                continue
            j = int(np.sum(self.fit_columns[fit_key].values[:i]))
            x = self.cumul[col][self.starts[fit_key][col]:self.end[fit_key][0]].values
            y = self.exp_fit(params, col, j, self.scales[fit_key], np.arange(lengths[i]))
            E += np.sum(np.abs(1-y/x)**2)
        return E
    
    def exp_fit(self, params, column, j, scale, t):
        m = int((np.size(params)-1)/2)
        return (scale[column][0]*params[j] + 
                scale[column][1]*params[m+j]*(np.exp(params[-1]*t)-1)/params[-1])
    
    def exp_fit_daily(self, params, column, j, scale, t):
        m = int((np.size(params)-1)/2)
        return (scale[column][1]*params[m+j]*np.exp(params[-1]*t))
    
    def fit_value_at(self, column, fit, at_date, daily = False):
        assert fit in self.params.columns
        assert self.fit_columns[fit][column]
        i = np.argwhere(self.columns == column)[0][0]
        j = int(np.sum(self.fit_columns[fit].values[:i]))
        start = date.datetime.strptime(self.starts[fit][column], self.date_format)
        d = date.datetime.strptime(at_date, self.date_format)
        t = (d-start).days
        params = self.params[fit].values[~np.isnan(self.params[fit])]
        if daily:
            return self.exp_fit_daily(params, column, j, self.scales[fit], t)
        else:
            return self.exp_fit(params, column, j, self.scales[fit], t)
    
    def date_to_time(self, dates):
        time = np.zeros(np.size(dates))
        for (i, d) in enumerate(dates):
            day = date.datetime.strptime(d, self.date_format)
            time[i] = (day-self.ref_datetime).days
        return time
    
    def plot(self, axes = None, fits = None):
        tick_interval = 25
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
            line, = axes.plot(self.date_to_time(self.daily[col].index), self.daily[col].values, 
                      linestyle = 'dashed', color = self.colors[i])
            data_lines.append(line)
        for (j, fit) in enumerate(fits):
            for (i, col) in enumerate(self.columns):
                if not self.fit_columns[fit][col]:
                    continue
                k = int(np.sum(self.fit_columns[fit].values[:i]))
                index = self.date_to_time(self.cumul[self.starts[fit][i]:self.end[fit][0]].index)
                params = self.params[fit].values[~np.isnan(self.params[fit])]
                line, = axes.plot(index, self.exp_fit_daily(self.params[fit].values, col, k,
                            self.scales[fit], np.arange(np.size(index))), 
                                  color = cm.jet(4*(self.params[fit].values[-1]+.1)))
                if k == 0:
                    line.set_label(fit + r': $\rho$ = %.1e' % self.params[fit][-1])
        axes.set_yscale('log')
        axes.legend(loc = 'best')
        axes.grid(True)
        axes.set_xticks(self.date_to_time(self.cumul.index[0::tick_interval]))
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
        
        
