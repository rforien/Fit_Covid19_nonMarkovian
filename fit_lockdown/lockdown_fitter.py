#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:39:13 2020

@author: raphael
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import datetime as date
import scipy.optimize as optim
import scipy.linalg as linalg
import scipy.special as sp
import time as timer
import copy

import fit_lockdown

def match_date_format(date_str, date_format = '%Y-%m-%d'):
    try:
        date.datetime.strptime(date_str, date_format)
        return True
    except ValueError:
        return False

def diff(date1, date2, date_format = '%Y-%m-%d'):
    d1 = date.datetime.strptime(date1, date_format)
    d2 = date.datetime.strptime(date2, date_format)
    return (d2-d1).days

def MGF_gamma(k, theta, rho):
    return (1+theta*rho)**(-k)

class Fit(object):
    def __init__(self, name, start, end, delays, columns = None):
        assert type(name)==str
        self.name = name
        assert match_date_format(start) and match_date_format(end)
        self.start = start
        self.end = end
        if columns:
            assert np.size(delays) == np.size(columns)
        assert np.min(delays) >= 0
        self.delays = delays
        self.columns = columns

class LockdownFitter(object):
    dpi = 200
    date_format = '%Y-%m-%d'
    
    default_delays = [10, 15, 15]
    start_fit_init = '2020-03-19'
    end_fit_init = '2020-03-27'
    columns_init = 'Hospital deaths'
    delays_init = [0]
    
    def __init__(self, data, name, size, lockdown_date):
        assert isinstance(data, pd.DataFrame)
        assert size > 0
        assert type(name) == str
        self.data = data
        self.events = data.columns
        self.name = name
        self.size = size
        self.datetime_lockdown = date.datetime.strptime(lockdown_date, self.date_format)
        
        self.fits = []
    
    def date_to_time(self, dates):
        time = np.zeros(np.size(dates))
        for (i, d) in enumerate(dates):
            date_i = date.datetime.strptime(d, self.date_format)
            time[i] = (date_i - self.datetime_lockdown).days
        return time
    
    def time_to_date(self, times):
        dates = np.size(times)*['']
        for (i, t) in enumerate(times):
            day = self.datetime_lockdown + date.timedelta(days = int(t))
            dates[i] = day.strftime(self.date_format)
        return dates
    
    def setup_fit(self, name, start, end, delays, columns = None):
        if not columns:
            assert np.size(delays) == np.size(self.events)
        if self.fits:
            assert diff(self.fits[-1].start, start) >= 0, "Trying to setup a fit before the last one."
        self.fits.append(Fit(name, start, end, delays, columns))
    
    def compute_growth_rates(self, verbose = False):
        print('Estimating growth rates in ' + self.name + '...')
        self.fitter = fit_lockdown.MultiFitter(self.data)
        assert len(self.fits) >= 2, "At least to fits are needed."
        for (i, fit) in enumerate(self.fits):
            print(fit.name)
            self.fitter.fit(fit.start, fit.end, fit.delays, fit.name, fit.columns, initial_phase=(i==0))
        self.rates = self.fitter.rates
        if verbose:
            print('Growth rates in ' + self.name)
            print(self.rates)
    
    def plot_fit(self, axs = None, francais = False, nb_xticks = 5):
        if not axs:
            fig = plt.figure(dpi = self.dpi)
            self.axs = plt.axes()
        else:
            self.axs = axs
            fig = None
        self.axs.set_title(self.name)
        data_lines = self.fitter.plot(self.axs, francais = francais, nb_xticks = nb_xticks)
        self.axs.legend(loc='best')
        if fig:
            fig.set_tight_layout(True)
        return data_lines
    
    def cumul_at_lockdown(self, event):
        init_phase = self.fits[0].name
        if self.fitter.in_fit[init_phase][event]:
            return self.fitter.fit_value_at(event, init_phase, self.datetime_lockdown.strftime(self.date_format))
        else:
            first_day = self.data[event].first_valid_index()
            cumul_first_day = self.data[event][first_day]
            difference = (date.datetime.strptime(first_day, self.date_format)-self.datetime_lockdown).days
            return cumul_first_day*np.exp(-self.rates[init_phase]*difference)
    
    def fit_delays(self, delta = 40):
        self.param_delays = pd.DataFrame(index = ['k', 'theta'])
        date_delta = (self.datetime_lockdown + date.timedelta(days = delta)).strftime(self.date_format)
        init_phase = self.fits[0].name
        second_phase = self.fits[1].name
        SIR_constants = self.sir.lockdown_constants(self.rates[second_phase], delta)
        EIR = self.sir.lead_eigen_vect(self.rates[init_phase])
        lambdaL = self.sir.contact_rate(self.rates[second_phase])
        K = np.sum(EIR*SIR_constants) - lambdaL*SIR_constants[0]*EIR[0]/self.rates[second_phase]
        for (i, event) in enumerate(self.events):
            cumul_at_lockdown = self.cumul_at_lockdown(event)
            cumul_at_delta = self.fitter.fit_value_at(event, second_phase, date_delta)
            daily_at_delta = self.fitter.fit_value_at(event, second_phase, date_delta, daily = True)
            A = (K*cumul_at_lockdown)/(cumul_at_delta - daily_at_delta/self.rates[second_phase])
            B = A*daily_at_delta/(cumul_at_lockdown*lambdaL*SIR_constants[0]*EIR[0])
            # print(A, B)
            propose = [7, 2]
            k, theta = propose
            t = 0
            while (k == propose[0] and theta == propose[1]) and t < 10:
                propose = np.random.uniform(1, 10, 2)
                k, theta = optim.fsolve(self._delay_err, propose, args = (A, B, self.rates[init_phase], self.rates[second_phase]))
                k = np.abs(k)
                theta = np.abs(theta)
                t += 1
            # print((1+self.r[i]*theta)**k, (1+self.rE[0,i]*theta)**k, k, theta)
            self.param_delays[event] = [k, theta]
          
    def _delay_err(self, X, A, B, r, rE):
        k, theta = np.abs(X)
        return [np.abs(MGF_gamma(k, theta, r)/A-1)**2, np.abs(MGF_gamma(k, theta, rE)/B-1)**2]
    
    def MGF_event(self, event, rho):
        return MGF_gamma(self.param_delays[event]['k'], self.param_delays[event]['theta'], rho)
    
    def compute_probas(self, p_ref, ref_event = 'Hospital deaths'):
        assert hasattr(self, 'param_delays')
        assert ref_event in self.events
        assert p_ref > 0 and p_ref < 1
        init_phase = self.fits[0].name
        self.probas = pd.Series()
        MGF_ref = self.MGF_event(ref_event, self.rates[init_phase])
        lockdown_ref = self.cumul_at_lockdown(ref_event)
        for (i, event) in enumerate(self.events):
            if event == ref_event:
                self.probas[event] = p_ref
            else:
                MGF = self.MGF_event(event, self.rates[init_phase])
                self.probas[event] = p_ref*(self.cumul_at_lockdown(event)*MGF_ref)/(lockdown_ref*MGF)
                assert self.probas[event] <= 1
    
    def prepare_sir(self, EI_dist, p_ref, ref_event = 'Hospital deaths', verbose = True):
        assert ref_event in self.events
        init_phase=self.fits[0].name
        self.sir = fit_lockdown.SEIR_nonMarkov(self.rates[init_phase], EI_dist, self.datetime_lockdown.strftime(self.date_format), self.size)
        self.fit_delays()
        self.compute_probas(p_ref, ref_event)
        if verbose:
            print('Mean delays in ' + self.name)
            for (i, event) in enumerate(self.events):
                print(event + ': %.3f' % np.prod(self.param_delays[event].values))
    
    def adjust_date_of_change(self, fit_name, event):
        assert hasattr(self, 'param_delays')
        assert event in self.events
        j = [fit.name for fit in self.fits].index(fit_name)
        assert j>0, 'Cannot adjust date of change for the first fit.'
        previous_fit = self.fits[j-1].name
        date1 = self.fits[j-1].start
        datetime1 = date.datetime.strptime(date1, self.date_format)
        date2 = self.fits[j].start
        datetime2 = date.datetime.strptime(date2, self.date_format)
        t1tot2 = (datetime2-datetime1).days
        if not hasattr(self, 'dates_of_change'):
            self.dates_of_change = {}
            for (i, fit) in enumerate(self.fits):
                if i > 0:
                    self.dates_of_change[fit.name] = fit.start
        rho1 = self.rates[previous_fit]
        R1 = self.fitter.fit_value_at(event, previous_fit, date1, daily = True)/self.MGF_event(event, rho1)
        rho2 = self.rates[fit_name]
        R2 = self.fitter.fit_value_at(event, fit_name, date2, daily = True)/self.MGF_event(event, rho2)
        t1totc = (rho2*t1tot2 + np.log(R1/R2))/(rho2-rho1)
        datetime_tc = datetime1 + date.timedelta(days = t1totc)
        self.dates_of_change[fit_name] = datetime_tc.strftime(self.date_format)
    
    def compute_intervals(self, dates_of_change, end_of_run):
        intervals = np.zeros(np.size(dates_of_change))
        date_end = date.datetime.strftime(end_of_run, self.date_format)
        for (j, d1) in enumerate(dates_of_change):
            d2 = np.concatenate((dates_of_change, [end_of_run]))[j+1]
            d2 = date.datetime.strptime(d2, self.date_format)
            if (d2-date_end).days <= 0:
                d2 = date_end
            d1 = date.datetime.strptime(d1, self.date_format)
            intervals[j] = np.maximum((d2-d1).days, 0)
        return intervals
    
    def build_delays(self):
        assert hasattr(self, 'param_delays')
        self.delays = {}
        for event in self.events:
            p = self.param_delays[event]
            self.delays[event] = fit_lockdown.gamma_dist(p['k'], p['theta'])
    
    def compute_sir(self, EI_dist, p_ref, end_of_run, ref_event = 'Hospital deaths', verbose = True, compute_events = True):
        if not hasattr(self, 'param_delays') or not hasattr(self, 'sir') or not hasattr(self, 'probas'):
            self.prepare_sir(EI_dist, p_ref, ref_event, verbose)
        assert ref_event in self.events
        if verbose:
            print('Running SEIR model in ' + self.name)
        if not hasattr(self, 'dates_of_change'):
            self.dates_of_change = {}
            for (i, fit) in enumerate(self.fits):
                if i > 0:
                    self.dates_of_change[fit.name] = fit.start
        intervals = self.compute_intervals([date for date in self.dates_of_change.values()], end_of_run)
        self.build_delays()
        self.sir.calibrate(self.cumul_at_lockdown(ref_event), p_ref, self.delays[ref_event])
        self.sir.run_up_to_lockdown(verbose = verbose)
        for (i, key) in enumerate(self.dates_of_change):
            if verbose:
                print(key)
            self.sir.change_contact_rate(self.rates[key], verbose = verbose)
            self.sir.run(intervals[i], record = True)
        if compute_events:
            for (j, event) in enumerate(self.events):
                self.sir.compute_delayed_event(self.probas[event], self.delays[event], event)
                
    def plot_delays(self, axs = None, legend = True):
        assert hasattr(self, 'param_delays')
        if not hasattr(self, 'delays'):
            self.build_delays()
        if not axs:
            fig = plt.figure(dpi = self.dpi)
            axs = plt.axes()
        axs.set_title(self.name)
        axs.set_xlabel('Days since infection')
        axs.set_xlim((0,40))
        for event in self.events:
            dx = np.concatenate(([1], np.diff(self.delays[event][:,0])))
            line, = axs.plot(self.delays[event][:,0], self.delays[event][:,1]/dx)
            mean = np.sum(self.delays[event][:,0]*self.delays[event][:,1])
            axs.vlines(mean, 0, np.max(self.delays[event][:,1]/dx), 
                       color = line.get_color(), linestyle = 'dashed')
            if legend:
                line.set_label('Infection to ' + event + ' delay')
        if legend:
            axs.legend(loc='best')
        if fig:
            fig.set_tight_layout(True)
    
    def plot_events(self, daily = True, logscale = False, nb_xticks = 8, axs = None):
        assert nb_xticks > 0
        tick_interval = int(np.size(self.sir.times)/nb_xticks)
        if not axs:
            fig = plt.figure(dpi = self.dpi)
            self.axs = plt.axes()
        else:
            self.axs = axs
        self.axs.set_title(self.name)
        self.axs.grid(True)
        if logscale:
            self.axs.set_yscale('log')
        if daily:
            data = self.data.diff()
            predictions = self.sir.daily
        else:
            data = self.data
            predictions = self.sir.cumul
        data_lines = self.axs.plot(self.date_to_time(self.data.index), data, 
                                     linestyle = 'dashed', linewidth = 1.2)
        sir_lines = self.axs.plot(self.sir.times-self.sir.lockdown_time, predictions)
        if logscale:
            self.axs.set_ylim((1e-1, 2*np.max(predictions.values)))
        for j in np.arange(np.size(data_lines)):
            sir_lines[j].set_color(data_lines[j].get_color())
        tick_times = (self.sir.times-self.sir.lockdown_time)[0::tick_interval]
        labels = self.time_to_date(tick_times)
        self.axs.set_xticks(tick_times)
        self.axs.set_xticklabels(labels)
        return data_lines, sir_lines