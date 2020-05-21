#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:06:28 2020

@author: raphael
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as date
import scipy.optimize as optim
import time

import fit_lockdown as lockdown

class FitPatches(object):
    lockdown_date = '2020-03-16'
    end_lockdown_fit = '2020-05-13'
    delay_deaths = 21
    delay_hosp = 13
    lockdown_end_date = '2020-05-11'
    date_format = '%Y-%m-%d'
    
    def __init__(self, deaths, hospitalisations, sizes):
        assert isinstance(deaths, pd.DataFrame) and isinstance(hospitalisations, pd.DataFrame)
        assert np.min(sizes) > 0
        self.n = np.size(sizes)
        self.sizes = sizes
        assert np.size(deaths.columns) == self.n and (hospitalisations.columns == deaths.columns).all()
        self.deaths = deaths
        self.hosp = hospitalisations
        self.names = deaths.columns
        self.datetime_lockdown = date.datetime.strptime(self.lockdown_date, self.date_format)
        self.datetime_end_lockdown = date.datetime.strptime(self.lockdown_end_date, self.date_format)
        self.lockdown_length = (self.datetime_end_lockdown - self.datetime_lockdown).days
        self.observed_times_deaths = self.index_to_time(deaths.index)
        self.observed_times_hosp = self.index_to_time(self.hosp.index)
    
    def index_to_time(self, index):
        time = np.zeros(np.size(index))
        for (i, d) in enumerate(index):
            date_i = date.datetime.strptime(d, self.date_format)
            time[i] = (date_i - self.datetime_lockdown).days
        return time
    
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
    
    def compute_sir(self, EI_dist, p_death, delay_death, delay_hosp, Markov = False):
        p_hosp = np.zeros(self.n)
        for (i, n) in enumerate(self.names):
            p_hosp[i] = p_death*(self.hosp[n].values[1]/self.deaths[n].values[1])*(
                    np.sum(delay_death[:,1]*np.exp(-self.r[i]*delay_death[:,0]))/
                  np.sum(delay_hosp[:,1]*np.exp(-self.r[i]*delay_hosp[:,0])))
        print('Probabilities of hospitalisation: ', p_hosp)
        self.sir = []
        if Markov:
            infectious_time = np.sum(EI_dist[:,1]*EI_dist[:,2])
            incubation_time = np.sum(EI_dist[:,0]*EI_dist[:,2])
            for i in np.arange(self.n):
                self.sir.append(lockdown.SEIR_lockdown_mixed_delays(self.sizes[i], self.r[i], self.rE[i], 
                                                                    p_death, incubation_time, 
                                                                    infectious_time, delay_death))
        else:
            for i in np.arange(self.n):
                self.sir.append(lockdown.SEIR_nonMarkov(self.sizes[i], self.r[i], self.rE[i], p_death, 
                                                                    EI_dist, delay_death))
        for (i, sir) in enumerate(self.sir):
            print('Running SEIR model in ' + self.names[i])
            sir.calibrate(self.deaths_at_lockdown[i], self.lockdown_date)
            sir.run_full(self.lockdown_length, 0, 1)
            sir.compute_deaths()
            sir.compute_hosp(p_hosp[i], delay_hosp)
            time.sleep(.001)
        
#    scale = np.array([1, 1, 1])
    def _fit_sir(self, params):
#        params = self.scale*params
#        params = np.abs(params)
#        params[0] = np.minimum(params[0], 1)
#        params[1:] = 10*params[1:]
        x = params[0]*np.array([.4, -.8, .4]) + params[1]*np.array([.7, 0, -.7])
        x = x + np.ones(3)/3
        x = np.maximum(x, np.zeros(3))
        x = np.minimum(x, np.ones(3))
        x = x/np.sum(x)
        print(x)
#        p_hosp = .025
        f = .005
        delay_death = np.transpose(np.vstack((np.linspace(11, 28, 10), .1*np.ones(10))))
        delay_hosp = np.transpose(np.vstack((np.linspace(6, 14, 10), .1*np.ones(10))))
        EI_dist = np.array([[3, 2, x[0]], [3, 7, x[1]], [3, 15, x[2]]])
        self.compute_sir(EI_dist, f, delay_death, delay_hosp)
        E = 0
        for (i, sir) in enumerate(self.sir):
            x = self.deaths[self.names[i]].values[:-5]
            dt = sir.times_death[1]-sir.times_death[0]
            y = sir.deaths[np.floor((self.observed_times_deaths[:-5]+sir.lockdown_time)/dt).astype(int)]
            E += np.mean((1-y/x)**2)
            x = self.hosp[self.names[i]].values[:-5]
            dt = sir.times_hosp[1]-sir.times_hosp[0]
            y = sir.hosp[np.floor((self.observed_times_hosp[:-5]+sir.lockdown_time)/dt).astype(int)]
            E += np.mean((1-y/x)**2)
        return E
    
    def fit_data(self, init_params):
        self.result = optim.minimize(self._fit_sir, init_params, method = 'Nelder-Mead')
        print(self.result.x)
    
    def plot_deaths_hosp(self):
        self.fig, self.dhaxs = plt.subplots(self.n, 2, dpi = 200, figsize = (8, 16), sharex = True)
        for (i, sir) in enumerate(self.sir):
            p0 = self.dhaxs[i,0].plot(sir.times_death-sir.lockdown_time, sir.deaths, label = 'predicted deaths')
            self.dhaxs[i,1].plot(sir.times_daily_deaths-sir.lockdown_time, sir.daily_deaths, color = p0[0].get_color())
            self.dhaxs[i,0].plot(self.observed_times_deaths, self.deaths[self.names[i]].values, 
                      label = 'observed_deaths', color = p0[0].get_color(), linestyle = 'dashed')
            self.dhaxs[i,1].plot(self.observed_times_deaths[1:], np.diff(self.deaths[self.names[i]].values),
                      color = p0[0].get_color(), linestyle = 'dashed')
            p1 = self.dhaxs[i,0].plot(sir.times_hosp-sir.lockdown_time, sir.hosp, label = 'predicted hospital admissions')
            self.dhaxs[i,1].plot(sir.times_daily_hosp-sir.lockdown_time, sir.daily_hosp, color = p1[0].get_color())
            self.dhaxs[i,0].plot(self.observed_times_hosp, self.hosp[self.names[i]].values, 
                      label = 'observed hospital admissions', color = p1[0].get_color(), linestyle = 'dashed')
            self.dhaxs[i,1].plot(self.observed_times_hosp[1:], np.diff(self.hosp[self.names[i]].values),
                      color = p1[0].get_color(), linestyle = 'dashed')
            self.dhaxs[i,1].set_yscale('log')
            self.dhaxs[i,0].set_yscale('log')
            self.dhaxs[i,0].set_xlim((-7, np.max(self.observed_times_deaths)))
            self.dhaxs[i,1].set_xlim((-7, np.max(self.observed_times_deaths)))
            self.dhaxs[i,0].set_ylim((10, 1e5))
            self.dhaxs[i,1].set_ylim((5, 1e4))
            self.dhaxs[i,0].set_ylabel(self.names[i])
            self.dhaxs[i,0].grid(True)
            self.dhaxs[i,1].grid(True)
        self.dhaxs[0,0].set_title('Cumulative data')
        self.dhaxs[0,1].set_title('Daily data')
        self.dhaxs[0,0].legend(loc = 'best')
        self.dhaxs[self.n-1,0].set_xlabel('Time since lockdown (days)')
        self.dhaxs[self.n-1,1].set_xlabel('Time since lockdown (days)')
    
    def plot_deaths_tot(self, deaths_tot):
        self.observed_times_deaths_tot = self.index_to_time(deaths_tot.index)
        plt.figure(dpi=200, figsize = (7,7))
        self.ax_tot = plt.axes()
        self.ax_tot.plot(self.observed_times_deaths_tot, deaths_tot.values, linestyle = 'dashed', 
                         label = 'observed deaths in France')
        shifts = [sir.shift(sir.lockdown_time) for sir in self.sir]
        j = np.argmax(shifts)
        shifts = np.max(shifts)-shifts
        deaths = np.zeros(np.max([np.size(sir.times_death) for sir in self.sir]))
        for (i, sir) in enumerate(self.sir):
            deaths[shifts[i]:] = deaths[shifts[i]:] + sir.deaths
        self.ax_tot.plot(self.sir[j].times_death-self.sir[j].lockdown_time, deaths, label = 'predicted deaths')
        self.ax_tot.legend(loc='best')
        self.ax_tot.set_yscale('log')
        self.ax_tot.set_xlim((-30,60))
        self.ax_tot.set_ylim((1, 2.5e4))
        self.ax_tot.set_xlabel('Time (days since lockdown)')
        self.ax_tot.set_ylabel('Cumulative hospital deaths')
        self.ax_tot.set_title('Predicted and observed hospital deaths using the 3-patches model')
        self.ax_tot.grid(True)
    
#    def _plot_deaths_hosp(self):
#        self.fig, self.dhaxs = plt.subplots(2, 2, dpi = 200, figsize = (8, 8), sharex = True)
#        for (i, sir) in enumerate(self.sir):
#            p = self.dhaxs[0,0].plot(self.observed_times_deaths, self.deaths[self.names[i]].values,
#                      label = self.names[i], linestyle = 'dashed')
#            color = p[0].get_color()
#            self.dhaxs[0,0].plot(sir.times_death-sir.lockdown_time, sir.deaths, color = color)
#            self.dhaxs[0,1].plot(self.observed_times_deaths[1:], np.diff(self.deaths[self.names[i]].values),
#                      linestyle = 'dashed', color = color)
#            self.dhaxs[0,1].plot(sir.times_daily_death-sir.lockdown_time, sir.daily_deaths, color = color)
#            self.dhaxs[1,0].plot(self.observed_times_hosp, self.hosp[self.names[i]].values, linestyle = 'dashed',
#                      color = color)
#            self.dhaxs[1,0].plot(sir.times_daily_hosp-sir.lockdown_time, sir.hosp, color = color)
#            self.dhaxs[1,1].plot(self.observed_times_hosp[1:], np.diff(self.hosp[self.names[i]].values),
#                      linestyle = 'dashed', color = color)
#            self.dhaxs[1,1].plot(sir.times_hosp-sir.lockdown_time, sir.daily_hosp, color = color)
#        self.dhaxs[0,0].set_title('Cumulative')
#        self.dhaxs[0,1].set_title('Daily')
#        self.dhaxs[0,0].set_ylabel('Deaths')
#        self.dhaxs[1,0].set_ylabel('Hospital admissions')
#        self.dhaxs[0,0].legend(loc='best')
#        self.dhaxs[1,0].set_xlabel('Time since lockdown (days)')
#        self.dhaxs[1,1].set_xlabel('Time since lockdown (days)')
##        self.dhaxs

