#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:06:28 2020

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
import time
import copy

#from .fit_lockdown import *
#from .sir import *
#from .dist import *

# import fit_lockdown
from fit_lockdown import *

class FitPatches(object):
    lockdown_date = '2020-03-16'
    lockdown_end_date = '2020-05-11'
    end_post_lockdown = '2020-06-16'
    dates_of_change = ['2020-03-16', '2020-05-11', '2020-06-02', '2020-07-10']
    dates_end_fit = ['2020-05-11', '2020-06-15', '2020-07-21', '2020-10-07']
    names_fit = ['Lockdown', 'After 11 May', 'After 2 June', 'After 10 July']
#    fit_columns = [None, None, ['Hospital admissions', 'Hospital deaths', 'SOS Medecins actions']]
#    delays = np.array([[18, 28, 28, 10], [10, 15, 15, 10], [15, 21, 10]])
    fit_columns = [None, None, None, None]
    # delays = np.array([[18, 28, 28, 10], [10, 15, 15, 8], [10, 15, 15, 8], [15, 21, 21, 10]])
    # fit_columns = [None, None, ['Hospital admissions']]
    # delays = np.array([[18, 28, 28, 10], [10, 15, 15, 8], [15]])
    delays = np.array([[18, 28, 28], [10, 15, 15], [10, 15, 15], [15, 21, 21]])
    # time to wait after lockdown to start fitting the slope
    delays_lockdown = np.array([18, 28, 28])
    # idem for post-lockdown fit
    delays_post = np.array([10, 15, 15])
    date_format = '%Y-%m-%d'
    start_fit_init = '2020-03-01'
    end_fit_init = '2020-03-22'
    
    date_first_measures_GE = '2020-03-07'
    r_GE = .27
    
    dpi = 200
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    def __init__(self, data, names, sizes):
        self.n = len(data)
        for i in np.arange(self.n):
            assert isinstance(data[i], pd.DataFrame)
        assert np.min(sizes) > 0 and np.size(sizes) == self.n
        assert np.size(names) == self.n
        self.data = data
        self.sizes = sizes
        self.names = names
        self.events = self.data[0].columns
        
        self.datetime_lockdown = date.datetime.strptime(self.lockdown_date, self.date_format)
        self.datetime_end_lockdown = date.datetime.strptime(self.lockdown_end_date, self.date_format)
        self.lockdown_length = (self.datetime_end_lockdown - self.datetime_lockdown).days
        self.observed_times = self.index_to_time(data[i].index)
    
    def index_to_time(self, index):
        time = np.zeros(np.size(index))
        for (i, d) in enumerate(index):
            date_i = date.datetime.strptime(d, self.date_format)
            time[i] = (date_i - self.datetime_lockdown).days
        return time
    
    def time_to_date(self, times):
        dates = np.size(times)*['']
        for (i, t) in enumerate(times):
            day = self.datetime_lockdown + date.timedelta(days = int(t))
            dates[i] = day.strftime(self.date_format)
        return dates
    
    def fit_patches(self, reference_column = 'Hospital deaths'):
        self.fitters = []
        self.death_fitters = []
        self.r = np.zeros(self.n)
        self.rE = np.zeros((np.size(self.dates_of_change), self.n))
        self.r_post = np.zeros(self.n)
        self.deaths_at_lockdown = np.zeros(self.n)
        for (i, n) in enumerate(self.names):
            print('Estimating growth rates in ' + n)
            self.fitters.append(MultiFitter(self.data[i]))
            self.death_fitters.append(Fitter(self.data[i][reference_column], self.lockdown_date, 0))
            self.death_fitters[i].fit_init(self.start_fit_init, self.end_fit_init)
            self.r[i] = self.death_fitters[i].r
            self.deaths_at_lockdown[i] = self.death_fitters[i].deaths_at_lockdown()
            for (j, fit) in enumerate(self.names_fit):
                self.fitters[i].fit(self.dates_of_change[j], self.dates_end_fit[j],
                            self.delays[j], fit, self.fit_columns[j])
                self.rE[j,i] = self.fitters[i].rates[fit]
        print('Growth rates prior to lockdown: ', self.r)
        for (j, name) in enumerate(self.names_fit):
            print('Growth rates ' + name, self.rE[j])
#        print('Deaths at lockdown: ', self.deaths_at_lockdown)
    
    def plot_fit_lockdown(self, display_legend = True, display_main_legend = True, francais = False):
        m = int(np.ceil(np.sqrt(self.n)))
        n = int(np.ceil(self.n/m))
        gs = gridspec.GridSpec(n, m)
        fig = plt.figure(dpi = self.dpi, figsize = (14, 8))
#        lines = []
        self.axs = []
        for i in np.arange(self.n):
            x = np.floor_divide(i, m)
            y = np.mod(i, m)
            if i == 0:
                self.axs.append(plt.subplot(gs[x, y]))
            else:
                self.axs.append(plt.subplot(gs[x, y], sharey = self.axs[-1]))
            self.axs[i].set_title(self.names[i])
            data_lines = self.fitters[i].plot(self.axs[i], francais = francais)
            index_init = self.fitters[i].date_to_time(self.death_fitters[i].index_init)
            if francais:
                label = r'avant le confinement : $t_2$ = %.1f jours' % (np.log(2)/self.r[i])
            else:
                label = r'Before lockdown : $t_2$ = %.1f days' % (np.log(2)/self.r[i])
            self.axs[i].plot(index_init, self.death_fitters[i].best_fit_init_daily(),
                    label = label, color = self.colors[0])
            if display_legend:
                self.axs[i].legend()
                # reorder legend (dirty)
                handles, labels = self.axs[i].get_legend_handles_labels()
                order = np.concatenate(([-1], np.arange(np.size(labels)-1)))
                handles = [handles[j] for j in order]
                labels = [labels[j] for j in order]
                print(labels)
                self.axs[i].legend(handles, labels, fontsize=13, loc = 'lower right')
        if display_main_legend:
            fig.legend(data_lines, self.fitters[0].columns, 
                       loc = (.9, .01), fontsize = 13)
        fig.set_tight_layout(True)
        return fig
        
    def fit_delays(self):
        delta = 40
        day_of_value = '2020-03-19'
        date_delta = (self.datetime_lockdown + date.timedelta(days = delta)).strftime(self.date_format)
        assert hasattr(self, 'fitters') and hasattr(self, 'sir')
        self.param_delays = pd.DataFrame(index = pd.MultiIndex.from_product([self.events, ['k', 'theta']]))
        for (i, fitter) in enumerate(self.fitters):
            sir = self.sir[i]
            SIR_constants = sir.lockdown_constants(self.rE[0,i], delta)
            EIR = sir.lead_eigen_vect(self.r[i])
            lambdaL = sir.contact_rate(self.rE[0,i])
            K = np.sum(EIR*SIR_constants) - lambdaL*SIR_constants[0]*EIR[0]/self.rE[0,i]
            day = date.datetime.strptime(day_of_value, self.date_format)
            diff = (day-self.datetime_lockdown).days
#            print(diff)
            self.param_delays[self.names[i]] = np.zeros(2*np.size(self.events))
            for (j, event) in enumerate(self.events):
                cumul_first_day = self.data[i][event][day_of_value]
                cumul_at_lockdown = cumul_first_day*np.exp(-self.r[i]*diff)
                first_fit = fitter.start.columns[0]
                cumul_at_delta = fitter.fit_value_at(event, first_fit, date_delta)
                daily_at_delta = fitter.fit_value_at(event, first_fit, date_delta, daily = True)
                A = (K*cumul_at_lockdown)/(cumul_at_delta - daily_at_delta/self.rE[0,i])
                B = A*daily_at_delta/(cumul_at_lockdown*lambdaL*SIR_constants[0]*EIR[0])
#                print(A, B)
                propose = [7, 2]
                k, theta = propose
                t = 0
                while (k == propose[0] and theta == propose[1]) and t < 10:
                    propose = np.random.uniform(1, 10, 2)
                    k, theta = optim.fsolve(self._delay_err, propose, args = (A, B, self.r[i], self.rE[0,i]))
                    k = np.abs(k)
                    theta = np.abs(theta)
                    t += 1
#                print((1+self.r[i]*theta)**k, (1+self.rE[0,i]*theta)**k, k, theta)
                self.param_delays[self.names[i]][event] = [k, theta]
        
    def _delay_err(self, X, A, B, r, rE):
        k, theta = np.abs(X)
        return [np.abs((1+r*theta)**(-k)/A-1)**2, np.abs((1+rE*theta)**(-k)/B-1)**2]
    
    def _delay_transform_tsm(self, r, r_before, tau, delay_dist):
        I1 = np.sum(np.exp(-r*delay_dist[:,0])*(delay_dist[:,0] <= tau)*delay_dist[:,1])
        I2 = np.exp(-(r-r_before)*tau)*np.sum(np.exp(-r_before*(delay_dist[:,0]))*(delay_dist[:,0] > tau)*delay_dist[:,1])
        return I1 + I2
    
    def build_delays(self):
        assert hasattr(self, 'param_delays')
        self.delays_events = []
        for (i, name) in enumerate(self.names):
            for (j, event) in enumerate(self.events):
                p = self.param_delays[name][event]
                self.delays_events.append(gamma_dist(p['k'], p['theta']))
    
    def compute_proba(self, p_death, event, i, reference_event = 'Hospital deaths'):
        at_date = '2020-03-20'
        j = int(np.argwhere(self.events == event)[0])
        k = int(np.argwhere(self.events == reference_event)[0])
        m = np.size(self.events)
        delay_ratio = (Laplace(self.delays_events[i*m+k], -self.r[i])/
                       Laplace(self.delays_events[i*m+j], -self.r[i]))
        return p_death*(self.data[i][event][at_date]/
                        self.data[i][reference_event][at_date])*delay_ratio
    
    def compute_proba_events(self, p_ref, ref_event = 'Hospital deaths'):
        self.probas = pd.DataFrame(columns = self.events, index = self.names)
        self.probas[ref_event] = p_ref*np.ones(self.n)
        for (j, event) in enumerate(self.events):
            if event == ref_event:
                continue
            self.probas[event] = np.zeros(self.n)
            for (i, name) in enumerate(self.names):
                self.probas[event][name] = self.compute_proba(p_ref, event, i, reference_event=ref_event)
                assert self.probas[event][name] > 0 and self.probas[event][name] <= 1
        
    
    # def compute_p_hosp(self, p_death):
    #     assert hasattr(self, 'delays_death') and hasattr(self, 'delays_hosp')
    #     self.p_hosp = np.zeros(self.n)
    #     for i in np.arange(self.n):
    #         if True:
    #             delay_ratio = Laplace(self.delays_death[i], -self.r[i])/Laplace(self.delays_hosp[i], -self.r[i])
    #         else:
    #             tau = self.sir[i].lockdown_time-self.sir[i].init_phase
    #             delay_ratio = (self._delay_transform_tsm(self.r[i], self.r_GE, tau, self.delays_death[i])/
    #                            self._delay_transform_tsm(self.r[i], self.r_GE, tau, self.delays_hosp[i]))
    #         self.p_hosp[i] = p_death*(self.data[i]['Hospital admissions']['2020-03-19']/
    #                        self.data[i]['Hospital deaths']['2020-03-19'])*delay_ratio
    
    def Laplace_event(self, patch, event, rho):
        return (1+self.param_delays[patch][event]['theta']*rho)**(-self.param_delays[patch][event]['k'])
    
    def adjust_dates_of_change(self, fit, event):
        j = self.names_fit.index(fit)
        assert j>0, 'Cannot adjust date of change for the first fit.'
        previous_fit = self.names_fit[j-1]
        date1 = self.dates_of_change[j-1]
        datetime1 = date.datetime.strptime(date1, self.date_format)
        date2 = self.dates_of_change[j]
        datetime2 = date.datetime.strptime(date2, self.date_format)
        t1tot2 = (datetime2-datetime1).days
        if not hasattr(self, 'adjusted_dates_of_change'):
            self.adjusted_dates_of_change = pd.DataFrame(index = self.names_fit)
            for (i, name) in enumerate(self.names):
                self.adjusted_dates_of_change[name] = self.dates_of_change
        for (i, name) in enumerate(self.names):
            rho1 = self.fitters[i].rates[previous_fit]
            R1 = self.fitters[i].fit_value_at(event, previous_fit, date1, daily = True)/self.Laplace_event(name, event, rho1)
            rho2 = self.fitters[i].rates[fit]
            R2 = self.fitters[i].fit_value_at(event, fit, date2, daily = True)/self.Laplace_event(name, event, rho2)
            t1totc = (rho2*t1tot2 + np.log(R1/R2))/(rho2-rho1)
            datetime_tc = datetime1 + date.timedelta(days = t1totc)
            self.adjusted_dates_of_change[name][fit] = datetime_tc.strftime(self.date_format)
    
    def prepare_sir(self, p_reported, p_ref, ref_event = 'Hospital deaths', Markov = False, verbose = True, two_step_measures = False):
        EI_dist = EI_dist_covid(p_reported)
        self.sir = []
        if Markov:
            print('Markov model')
            infectious_time = np.sum(EI_dist[:,1]*EI_dist[:,2])
            incubation_time = np.sum(EI_dist[:,0]*EI_dist[:,2])
            for i in np.arange(self.n):
                self.sir.append(SEIR_lockdown(self.r[i], infectious_time, incubation_time, self.lockdown_date,
                                              self.sizes[i], two_step_measures = (i==1 and two_step_measures),
                                              growth_rate_before_measures = self.r_GE, 
                                              date_of_measures = self.date_first_measures_GE))
        else:
            for i in np.arange(self.n):
                self.sir.append(SEIR_nonMarkov(self.r[i], EI_dist, self.lockdown_date, self.sizes[i],
                                               two_step_measures = (i==1 and two_step_measures),
                                               growth_rate_before_measures = self.r_GE, 
                                               date_of_measures = self.date_first_measures_GE))
        self.fit_delays()
        self.build_delays()
        self.compute_proba_events(p_ref, ref_event)
#        self.delays_hosp = []
#        self.delays_icu = []
#        self.delays_death = []
#        for i in np.arange(self.n):
#            for (j, delays) in enumerate([self.delays_hosp, self.delays_death, self.delays_icu]):
#                p = self.param_delays[self.names[i]][self.events[j]]
#                delays.append(gamma_dist(p['k'], p['theta']))
#        self.compute_p_hosp(p_death)
        if verbose:
            for (i, name) in enumerate(self.names):
                for (j, event) in enumerate(self.events):
                    print('Mean infection to ' + event + ' delay in ' + name + ': ', np.prod(self.param_delays[name][event].values))
            # print('Probabilities of hospitalisation: ', self.probas['Hospital admissions'])
#            print('relative probabilities: ', p_death/self.probas['Hospital admissions'])

    def compute_intervals(self, dates_of_change, end_of_run):
        intervals = np.zeros(np.size(dates_of_change))
        for (j, d1) in enumerate(dates_of_change):
            d2 = np.concatenate((dates_of_change, [end_of_run]))[j+1]
            d1 = date.datetime.strptime(d1, self.date_format)
            d2 = date.datetime.strptime(d2, self.date_format)
            intervals[j] = np.maximum((d2-d1).days, 0)
        return intervals
            
    def compute_sir(self, p_reported, p_ref, end_of_run, ref_event = 'Hospital deaths', Markov = False, verbose = True, two_step_measures = False):
        if not hasattr(self, 'param_delays'):
            self.prepare_sir(p_reported, p_ref, ref_event, Markov, verbose, two_step_measures)
        
        m = np.size(self.events)
        assert ref_event in self.events
        k = int(np.argwhere(self.events == ref_event)[0])
        for (i, sir) in enumerate(self.sir):
            if verbose:
                print('Running SEIR model in ' + self.names[i])
            if hasattr(self, 'adjusted_dates_of_change'):
                intervals = self.compute_intervals(self.adjusted_dates_of_change[self.names[i]], end_of_run)
            else:
                intervals = self.compute_intervals(self.dates_of_change, end_of_run)
            sir.calibrate(self.deaths_at_lockdown[i], p_ref, self.delays_events[i*m+k])
            sir.run_up_to_lockdown(verbose = verbose)
            for (j, name) in enumerate(self.names_fit):
                if name == 'Before lockdown':
                    continue
                if verbose:
                    print(name)
                sir.change_contact_rate(self.rE[j,i], verbose = verbose)
                sir.run(intervals[j], record = True)
            for (j, event) in enumerate(self.events):
                sir.compute_delayed_event(self.probas[event][self.names[i]],
                                           self.delays_events[i*m+j], event)
            time.sleep(.001)
    
    def plot_delays(self):
        m = int(np.ceil(np.sqrt(self.n)))
        n = int(np.ceil(self.n/m))
        p = len(self.events)
        gs = gridspec.GridSpec(n, m)
        fig = plt.figure(dpi = self.dpi, figsize = (12, 7))
        self.axs = []
        for (i, name) in enumerate(self.names):
            x = np.floor_divide(i, m)
            y = np.mod(i, m)
            self.axs.append(plt.subplot(gs[x, y]))
            self.axs[i].set_title(name)
            self.axs[i].set_xlabel('Days since infection')
            self.axs[i].set_xlim((0, 40))
            for (j, event) in enumerate(self.events):
                dx = np.concatenate(([1], np.diff(self.delays_events[i*p+j][:,0])))
                line, = self.axs[i].plot(self.delays_events[i*p+j][:,0], self.delays_events[i*p+j][:,1]/dx)
                mean = np.sum(self.delays_events[i*p+j][:,0]*self.delays_events[i*p+j][:,1])
                self.axs[i].vlines(mean, 0, np.max(self.delays_events[i*p+j][:,1]/dx), 
                                   color = line.get_color(), linestyle = 'dashed')
                if i==self.n-1:
                    line.set_label('Infection to ' + event + ' delay')
            if i==self.n-1:
                self.axs[i].legend(loc='best')
        fig.set_tight_layout(True)
        
    def plot_events(self, daily = True, logscale = True):
        tick_interval = int(np.size(self.sir[0].times)/8)
        print(tick_interval)
        m = int(np.ceil(np.sqrt(self.n)))
        n = int(np.ceil(self.n/m))
        gs = gridspec.GridSpec(n, m)
        fig = plt.figure(dpi = self.dpi, figsize = (12, 7))
#        lines = []
        self.axs = []
        for (i, sir) in enumerate(self.sir):
            x = np.floor_divide(i, m)
            y = np.mod(i, m)
            self.axs.append(plt.subplot(gs[x, y]))
            self.axs[i].set_title(self.names[i])
            self.axs[i].grid(True)
            if logscale:
                self.axs[i].set_yscale('log')
            
            if daily:
                data_lines = self.axs[i].plot(self.index_to_time(self.data[i].index), self.data[i].diff(), 
                                         linestyle = 'dashed', linewidth = 1.2)
                sir_lines = self.axs[i].plot(sir.times-sir.lockdown_time, sir.daily)
                if logscale:
                    self.axs[i].set_ylim((1e-1, 2*np.max(sir.daily.values)))
            else:
                data_lines = self.axs[i].plot(self.index_to_time(self.data[i].index), self.data[i],
                                              linestyle = 'dashed', linewidth = 1.2)
                sir_lines = self.axs[i].plot(sir.times-sir.lockdown_time, sir.cumul)
                if logscale:
                    self.axs[i].set_ylim((1e-1, 2*np.max(sir.cumul.values)))
            for j in np.arange(np.size(data_lines)):
                sir_lines[j].set_color(data_lines[j].get_color())
            tick_times = (sir.times-sir.lockdown_time)[0::tick_interval]
            labels = self.time_to_date(tick_times)
            self.axs[i].set_xticks(tick_times)
            self.axs[i].set_xticklabels(labels)
        labels = []
        for event in self.events.values:
            labels.append('Prédictions')
        fig.legend(tuple(data_lines + sir_lines), tuple(self.events) + tuple(labels), 
                    loc = (.1, .75), ncol = 2, fontsize = 12)
        fig.set_tight_layout(True)
    
    def plot_deaths_hosp(self, logscale = True):
        self.fig, self.dhaxs = plt.subplots(self.n, 2, dpi = self.dpi, figsize = (10, 16), sharex = True)
        for (i, sir) in enumerate(self.sir):
            p0 = self.dhaxs[i,0].plot(sir.times-sir.lockdown_time, sir.cumul['Hospital deaths'].values, 
                           label = 'predicted deaths')
            self.dhaxs[i,1].plot(sir.times-sir.lockdown_time, sir.daily['Hospital deaths'].values, 
                      color = p0[0].get_color())
            self.dhaxs[i,0].plot(self.observed_times, self.data[i]['Hospital deaths'].values, 
                      label = 'observed_deaths', color = p0[0].get_color(), linestyle = 'dashed')
            self.dhaxs[i,1].plot(self.observed_times[1:], np.diff(self.data[i]['Hospital deaths'].values),
                      color = p0[0].get_color(), linestyle = 'dashed')
            p1 = self.dhaxs[i,0].plot(sir.times-sir.lockdown_time, sir.cumul['Hospital admissions'].values, 
                           label = 'predicted hospital admissions')
            self.dhaxs[i,1].plot(sir.times-sir.lockdown_time, sir.daily['Hospital admissions'].values, 
                      color = p1[0].get_color())
            self.dhaxs[i,0].plot(self.observed_times, self.data[i]['Hospital admissions'].values, 
                      label = 'observed hospital admissions', color = p1[0].get_color(), linestyle = 'dashed')
            self.dhaxs[i,1].plot(self.observed_times[1:], np.diff(self.data[i]['Hospital admissions'].values),
                      color = p1[0].get_color(), linestyle = 'dashed')
            if logscale:
                self.dhaxs[i,1].set_yscale('log')
                self.dhaxs[i,0].set_yscale('log')
                self.dhaxs[i,0].set_xlim((-7, np.max(self.observed_times)))
                self.dhaxs[i,1].set_xlim((-7, np.max(self.observed_times)))
                self.dhaxs[i,0].set_ylim((10, 1e5))
                self.dhaxs[i,1].set_ylim((1, 1e4))
            self.dhaxs[i,0].set_ylabel(self.names[i])
            self.dhaxs[i,0].grid(True)
            self.dhaxs[i,1].grid(True)
        self.dhaxs[0,0].set_title('Cumulative data')
        self.dhaxs[0,1].set_title('Daily data')
        self.fig.legend()
        self.dhaxs[self.n-1,0].set_xlabel('Time since lockdown (days)')
        self.dhaxs[self.n-1,1].set_xlabel('Time since lockdown (days)')
        for i in np.arange(self.n):
            for j in np.arange(2):
                times = self.dhaxs[i,j].get_xticks()
                labels = self.time_to_date(times)
                self.dhaxs[i, j].set_xticklabels(labels)
    
#    def plot_deaths_tot(self, deaths_tot):
#        self.observed_times_deaths_tot = self.index_to_time(deaths_tot.index)
#        plt.figure(dpi=self.dpi, figsize = (7,7))
#        self.ax_tot = plt.axes()
#        self.ax_tot.plot(self.observed_times_deaths_tot, deaths_tot.values, linestyle = 'dashed', 
#                         label = 'observed deaths in France')
#        shifts = [sir.shift(sir.lockdown_time) for sir in self.sir]
#        j = np.argmax(shifts)
#        shifts = np.max(shifts)-shifts
#        deaths = np.zeros(np.max([np.size(sir.times_death) for sir in self.sir]))
#        for (i, sir) in enumerate(self.sir):
#            deaths[shifts[i]:] = deaths[shifts[i]:] + sir.deaths
#            self.ax_tot.plot(sir.times_death-sir.lockdown_time, sir.deaths, label = self.names[i])
#        self.ax_tot.plot(self.sir[j].times_death-self.sir[j].lockdown_time, deaths, 
#                         label = 'predicted deaths in France')
#        self.ax_tot.legend(loc='best')
#        self.ax_tot.set_yscale('log')
#        self.ax_tot.set_xlim((-30,60))
#        self.ax_tot.set_ylim((1, 2.5e4))
#        self.ax_tot.set_xlabel('Time (days since lockdown)')
#        self.ax_tot.set_ylabel('Cumulative hospital deaths')
#        self.ax_tot.set_title('Predicted and observed hospital deaths using the 3-patches model')
#        self.ax_tot.grid(True)
    
    def compute_deaths_tot(self):
        assert hasattr(self, 'sir')
        shifts = [sir.shift(sir.lockdown_time) for sir in self.sir]
        j = np.argmax(shifts)
        shifts = np.max(shifts)-shifts
        self.deaths_tot = np.zeros(np.max([np.size(sir.times) for sir in self.sir]))
        for (i, sir) in enumerate(self.sir):
            self.deaths_tot[shifts[i]:] = self.deaths_tot[shifts[i]:] + sir.cumul['Hospital deaths'].values
        self.times_deaths_tot = self.sir[j].times-self.sir[j].lockdown_time
        
    
    def plot_fit_init(self, deaths_tot, p_reported, p_death):
        self.tot_fitter = Fitter(deaths_tot, self.lockdown_date, 1)
#        self.tot_fitter.fit_init('2020-03-01', self.end_fit_init)
        self.observed_times_tot = self.index_to_time(self.tot_fitter.data.index)
#        self.interval_fit = self.index_to_time(self.tot_fitter.index_init)
        
        self.fig, self.init_axs = plt.subplots(1, 2, dpi = self.dpi, figsize = (9, 5))
        self.init_axs[0].set_ylabel('Cumulative number of hospital deaths')
        self.init_axs[0].set_title('Without two-step measures')
        self.init_axs[1].set_title('With two-step measures in Grand Est\nand Hauts-de-France')
        
        # plot data
        for i in np.arange(2):
            self.init_axs[i].plot(self.observed_times_tot, self.tot_fitter.data['cumul'].values, 
                                  linestyle = 'dashed', linewidth = 1.2, color = self.colors[0])
#            self.init_axs[i].plot(self.interval_fit, self.tot_fitter.best_fit_init_cumul(), 
#                                  color = self.colors[0], linestyle = 'dashdot')
            for j in np.arange(self.n):
                self.init_axs[i].plot(self.index_to_time(self.death_fitters[j].data.index), 
                                      self.death_fitters[j].data['cumul'].values,
                                      linestyle = 'dashed', linewidth = 1.2, color = self.colors[1+j])
#                self.init_axs[i].plot(self.index_to_time(self.death_fitters[j].index_init), 
#                                      self.death_fitters[j].best_fit_init_cumul(),
#                                      color = self.colors[1+i], linestyle = 'dashdot')
            self.init_axs[i].set_yscale('log')
            self.init_axs[i].grid(True)
        
        # plot sir with two step measures
        self.compute_sir(p_reported, p_death, '2020-05-30')
        self.compute_deaths_tot()
        self.init_axs[1].plot(self.times_deaths_tot, self.deaths_tot, linewidth = 1.5, color = self.colors[0])
        for (i, sir) in enumerate(self.sir):
            self.init_axs[1].plot(sir.times-sir.lockdown_time, sir.cumul['Hospital deaths'].values, 
                                  linewidth = 1.5, color = self.colors[1+i])
        
        # plot sir without two step measures
        self.compute_sir(p_reported, p_death, '2020-05-30', two_step_measures = False)
        self.compute_deaths_tot()
        self.init_axs[0].plot(self.times_deaths_tot, self.deaths_tot, linewidth = 1.5, color = self.colors[0], 
                     label = 'France')
        for (i, sir) in enumerate(self.sir):
            self.init_axs[0].plot(sir.times-sir.lockdown_time, sir.cumul['Hospital deaths'].values, 
                         linewidth = 1.5, color = self.colors[1+i], label = self.names[i])
        self.init_axs[0].legend(loc = 'best')
        
        for i in np.arange(2):
            self.init_axs[i].set_ylim((.5, 4e4))
            self.init_axs[i].set_xlim((-40, 40))
            self.init_axs[i].set_xlabel('Time (days since lockdown)')
        self.fig.set_tight_layout(True)
    
#    def plot_R0(self):
#        n = 100
#        p_reported = np.linspace(0, 1, n)
#        R0_pre = np.zeros((n, self.n))
#        R0_lock = np.zeros((n, self.n))
#        for i in np.arange(n):
#            EI_dist = EI_dist_covid(p_reported[i])
#            for j in np.arange(self.n):
#                R0_pre[i, j] = R0(self.r[j], EI_dist)
#                R0_lock[i, j] = R0(self.rE[j], EI_dist)
#        plt.figure(dpi = self.dpi, figsize = (6, 6))
#        self.axs = plt.axes()
#        for i in np.arange(self.n):
#            self.axs.plot(p_reported, R0_pre[:,i], label = '$R_0$ before lockdown in ' + self.names[i])
#            self.axs.plot(p_reported, R0_lock[:,i], label = '$R_0$ during lockdown in ' + self.names[i])
#        self.axs.legend(loc = 'best')
#        self.axs.set_yscale('log')
#        self.axs.set_xlabel('Proportion of reported individuals')
    
    linestyles = ['dashed', 'solid', 'dashdot']
    
    def plot_immunity(self, f_values, p_reported, end_date, logscale = False):
        self.fig, self.axs = plt.subplots(1, self.n, dpi = self.dpi, figsize = (12, 4), sharey = True, sharex = True)
        if self.n == 1:
            self.axs = [self.axs]
        for i in np.arange(self.n):
            self.axs[i].set_title(self.names[i])
#            self.axs[i].set_xlabel('Time (days since lockdown)')
            self.axs[i].grid(True)
            self.axs[i].yaxis.set_tick_params(which = 'both', labelleft = True)
            if logscale:
                self.axs[i].set_yscale('log')
        self.axs[0].set_ylabel('Proportion of infected individuals (1-S)', fontsize = 11)
        
        for (j, f) in enumerate(f_values):
            self.compute_sir(p_reported, f, end_date, two_step_measures = False)
            for (i, sir) in enumerate(self.sir):
                self.axs[i].plot(sir.times-sir.lockdown_time, 1-sir.traj[:,0],
                        label = 'f = %.2f%%' % (100*f), 
                        linestyle = self.linestyles[np.mod(j, np.size(self.linestyles))])
        self.axs[-1].legend(loc = 'upper right', title = 'Infection fatality ratio')
        
        tick_interval = int(np.size(self.sir[0].times)/2)
        for (i, sir) in enumerate(self.sir):
            tick_times = (sir.times-sir.lockdown_time)[0::tick_interval]
            labels = self.time_to_date(tick_times)
            self.axs[i].set_xticks(tick_times)
            self.axs[i].set_xticklabels(labels)
        self.fig.set_tight_layout(True)
    
    def plot_SIR_deaths_hosp(self, logscale = True):
        assert hasattr(self, 'sir')
        self.fig, self.faxs = plt.subplots(self.n, 3, dpi = self.dpi, figsize = (12, 12), sharex = False)
        self.faxs[0, 0].set_title('trajectory of the SEIR model')
        self.faxs[0, 1].set_title('Hospital admissions')
        self.faxs[0, 2].set_title('Hospital deaths')
        for i in np.arange(3):
            self.faxs[self.n-1, i].set_xlabel('Time (days since lockdown)')
        ymax = np.max([1-sir.traj[-1,0] for sir in self.sir])+.05
        xmax = np.max([sir.times[-1]-sir.lockdown_time for sir in self.sir])
        for (i, sir) in enumerate(self.sir):
            self.faxs[i,0].set_ylabel(self.names[i])
            self.faxs[i,0].plot(sir.times-sir.lockdown_time, 1-sir.traj[:,0], label = '1-S')
            self.faxs[i,0].plot(sir.times-sir.lockdown_time, sir.traj[:,3], label = 'E')
            self.faxs[i,0].plot(sir.times-sir.lockdown_time, sir.traj[:,1], label = 'I')
            self.faxs[i,0].plot(sir.times-sir.lockdown_time, sir.traj[:,2], label = 'R')
            self.faxs[i,0].grid(True)
            self.faxs[i,0].vlines(0, 0, ymax, linestyle = 'dashed')
            self.faxs[i,0].legend(loc='best')
            self.faxs[i,0].set_ylim((-.01, ymax))
            
            for (j, name) in enumerate(['Hospital admissions', 'Hospital deaths']):
                cumul = sir.plot_event(name, axes = self.faxs[i,1+j], labels = False)
                cumul.set_label('Cumulative')
                daily = sir.plot_event(name, daily = True, axes = self.faxs[i,1+j], labels = False)
                daily.set_label('Daily')
                self.faxs[i,1+j].plot(self.observed_times, self.data[i][name].values, 
                         linestyle = 'dashed', color = cumul.get_color())
                self.faxs[i,1+j].plot(self.observed_times[1:], np.diff(self.data[i][name].values),
                         linestyle = 'dashed', color = daily.get_color())
                if logscale:
                    self.faxs[i,1+j].set_yscale('log')
                self.faxs[i,1+j].legend(loc='best')
                self.faxs[i,1+j].set_xlim((-7, np.maximum(np.max(self.observed_times), xmax)))
            
            if logscale:
                self.faxs[i,1].set_ylim((1e0, 5e5))
                self.faxs[i,2].set_ylim((1e-1, 1e5))
        for i in np.arange(self.n):
            for j in np.arange(3):
                times = self.faxs[i, j].get_xticks()
                labels = self.time_to_date(copy.copy(times))
#                print(times, labels)
                self.faxs[i, j].set_xticklabels(copy.copy(labels))
        self.fig.set_tight_layout(True)
    
    def plot_markov_vs_nonmarkov(self, p_reported, p_death, logscale = False):
        self.fig, self.axs = plt.subplots(self.n, 3, dpi = self.dpi, figsize = (11, 9))
        self.axs[0,0].set_title('Proportion of infected individuals (1-S)')
        self.axs[0,1].set_title('Daily hospital admissions')
        self.axs[0,2].set_title('Daily hospital deaths')
        for i in np.arange(3):
            self.axs[self.n-1, i].set_xlabel('Time (days since lockdown)')
        for i in np.arange(self.n):
            self.axs[i,0].set_ylabel(self.names[i], fontsize = 12)
            if not logscale:
                self.axs[i,0].set_ylim((-.01, .2))
            else:
                self.axs[i,0].set_ylim((1e-2, 2e-1))
            for j in np.arange(3):
                self.axs[i,j].grid(True)
                if logscale:
                    self.axs[i,j].set_yscale('log')
                    if j > 0:
                        self.axs[i,j].set_ylim((1, 3*10**(4-j)))
            
        # plot data
        for (i, fitter) in enumerate(self.fitters):
            for (k, obs) in enumerate(['Hospital admissions', 'Hospital deaths']):
                self.axs[i,1+k].plot(self.observed_times, fitter.daily[obs].values,
                        linestyle = 'dashed', color = self.colors[k], linewidth = 1.2)
        
        self.compute_sir(p_reported, p_death, self.data[0].index[-1])
        # plot non-Markov
        for (i, sir) in enumerate(self.sir):
            line, = self.axs[i,0].plot(sir.times-sir.lockdown_time, 1-sir.traj[:,0], 
                    color = self.colors[2])
            if i == 0:
                line.set_label('non-Markovian SEIR model')
            for (j, name) in enumerate(['Hospital admissions', 'Hospital deaths']):
                sir.plot_event(name, daily = True, axes = self.axs[i,1+j], labels = False,
                               color = self.colors[2], linewidth = 1.5)
        
        self.compute_sir(p_reported, p_death, self.data[0].index[-1], Markov = True)
        # plot Markov
        for (i, sir) in enumerate(self.sir):
            line, = self.axs[i,0].plot(sir.times-sir.lockdown_time, 1-sir.traj[:,0],
                    linestyle = 'dashdot', color = self.colors[3])
            if i == 0:
                line.set_label('Markovian SEIR model')
            for (j, name) in enumerate(['Hospital admissions', 'Hospital deaths']):
                sir.plot_event(name, daily = True, axes = self.axs[i,1+j], labels = False,
                               color = self.colors[3], linewidth = 1.5)
        self.fig.legend(loc = 'lower right', fontsize = 12)
        self.fig.subplots_adjust(left = .07, right = .97, top = .95, bottom = .15, hspace = 0.2, wspace = .2)
            
    '''
    def concat_sir(self):
        assert hasattr(self, 'sir')
        dt = [sir.times[1]-sir.times[0] for sir in self.sir]
        assert (np.diff(dt) == np.zeros(self.n-1)).all()
        shifts = [sir.shift(sir.lockdown_time) for sir in self.sir]
        j = np.argmax(shifts)
        shifts = np.max(shifts)-shifts
        # renormalisation factors to account for different sizes
        p = self.sizes/np.sum(self.sizes)
        self.times = self.sir[j].times
        n = np.size(self.times)
        self.traj = np.zeros((n, self.n, self.sir[0].n))
        self.Z = np.zeros((self.n, self.sir[0].n))
        if hasattr(self.sir[0], 'flux'):
            m = np.size(self.sir[j].flux, axis = 0)
            self.flux = np.zeros((m, self.n, self.sir[0].n-1))    
        for (i, sir) in enumerate(self.sir):
            self.traj[shifts[i]:,i,:] = p[i]*sir.traj
            self.traj[:shifts[i],i,0] = p[i]*np.ones(shifts[i])
            if hasattr(sir, 'flux'):
                self.flux[shifts[i]:,i,:] = p[i]*sir.flux
            self.Z[i,:] = p[i]*sir.Z
        for i in np.arange(self.n):
            self.sir[i].times = self.times
            self.sir[i].traj = self.traj[:,i,:]
            self.sir[i].lockdown_time = self.sir[j].lockdown_time
            self.sir[i].Z = self.Z[i,:]
            if hasattr(self.sir[i], 'flux'):
                self.sir[i].flux = self.flux[:,i,:]
                self.sir[i].i = n-1
#        self.i = n
    
    def run_patches(self, T, MigMat, dt = .01):
        if not hasattr(self, 'traj'):
            self.concat_sir()
        assert dt == self.times[1]-self.times[0]
        n = int(T/dt)
        self.p = self.sir[0].n
        last_t = self.times[-1]
        offset = np.size(self.times)
        self.times = np.concatenate((self.times, last_t + dt*np.arange(1,n+1)))
        self.traj = np.concatenate((self.traj, np.zeros((n, self.n, self.p))), axis = 0)
        if hasattr(self.sir[0], 'flux'):
            self.flux = np.concatenate((self.flux, np.zeros((n, self.n, self.p-1))), axis = 0)
            self.window_size = np.size(self.flux, axis = 0)-np.size(self.times)
            self.index_flux = np.zeros(self.p-1)
            for i in np.arange(self.p-1):
                self.index_flux[i] = np.where(self.sir[0].A[:,i] == -1)[0][0]
        for i in np.arange(self.n):
            self.sir[i].times = self.times
            self.sir[i].traj = self.traj[:,i,:]
            # update lockdown time !!
            if hasattr(self.sir[i], 'flux'):
                self.sir[i].flux = self.flux[:,i,:]
                self.sir[i].i = offset-1
        self.exp_migmat = np.zeros((self.p, self.n, self.n))
        for j in np.arange(self.p):
            assert np.shape(MigMat[j]) == (self.n, self.n) and np.prod([np.sum(MigMat[j,i,:]) == 0 for i in np.arange(self.n)])
            self.exp_migmat[j] = linalg.expm(dt*MigMat[j])
        for self.t in np.arange(offset, offset+n):
            self._step(dt)
            self.traj[self.t] = self.Z
#            for (i, sir) in enumerate(self.sir):
#                self.traj[offset+t,i,:] = sir.Z
    
    def _step(self, dt):
        for (i, sir) in enumerate(self.sir):
            sir._step(dt)
            self.Z[i,:] = sir.Z
        for j in np.arange(self.p):
            self.Z[:,j] = np.matmul(self.exp_migmat[j], self.Z[:,j])
            # problem with indices between flux and migmat
        for i in np.arange(self.n):
            self.sir[i].Z = self.Z[i,:]
        if hasattr(self.sir[0], 'flux'):
            for j in np.arange(1,self.p-1):
                self.flux[self.t:(self.t+self.window_size),:,j] = np.transpose(np.matmul(
                        self.exp_migmat[int(self.index_flux[j])], 
                         np.transpose(self.flux[self.t:(self.t+self.window_size),:,j])))
'''

