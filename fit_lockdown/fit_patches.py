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
import datetime as date
import scipy.optimize as optim
import scipy.linalg as linalg
import scipy.special as sp
import time

#from .fit_lockdown import *
#from .sir import *
#from .dist import *

# import fit_lockdown
from fit_lockdown import *

class FitPatches(object):
    lockdown_date = '2020-03-16'
    end_lockdown_fit = '2020-05-13'
    # time to wait after lockdown to start fitting the slope
    delay_deaths = 25
    delay_hosp = 18
    lockdown_end_date = '2020-05-11'
    date_format = '%Y-%m-%d'
    start_fit_init = '2020-03-19'
    end_fit_init = '2020-03-26'
    start_post_lockdown_fit = '2020-05-21'
    end_post_lockdown_fit = '2020-06-10'
    
    date_first_measures_GE = '2020-03-07'
    r_GE = .27
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
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
        self.r_post = np.zeros(self.n)
        self.deaths_at_lockdown = np.zeros(self.n)
        for (i, n) in enumerate(self.names):
            self.death_fitters.append(Fitter(self.deaths[n], self.lockdown_date, self.delay_deaths))
            self.death_fitters[i].fit_init(self.start_fit_init, self.end_fit_init)
            self.death_fitters[i].fit_lockdown(self.end_lockdown_fit)
            self.death_fitters[i].fit_post_lockdown('2020-05-25', self.end_post_lockdown_fit)
            self.hosp_fitters.append(Fitter(self.hosp[n], self.lockdown_date, self.delay_hosp))
            self.hosp_fitters[i].fit_lockdown(self.end_lockdown_fit)
            self.hosp_fitters[i].fit_post_lockdown(self.start_post_lockdown_fit, self.end_post_lockdown_fit)
            self.r[i] = self.death_fitters[i].r
            self.rE[i] = self.hosp_fitters[i].rE
            self.r_post[i] = self.hosp_fitters[i].r_post
            self.deaths_at_lockdown[i] = self.death_fitters[i].deaths_at_lockdown()
        print('Growth rates prior to lockdown: ', self.r)
        print('Growth rates during lockdown: ', self.rE)
        print('Growth rates after lockdown: ', self.r_post)
        print('Deaths at lockdown: ', self.deaths_at_lockdown)
    
    def plot_fit(self):
        tick_interval = 21
        self.fig, self.axs = plt.subplots(self.n, 2, dpi = 200, figsize = (10, 16), sharex = True)
        self.axs[0,0].set_title('Cumulative data')
        self.axs[0,1].set_title('Daily data')
        for i in np.arange(self.n):
            self.axs[i,0].set_ylabel(self.names[i], fontsize = 12)
            deaths = self.axs[i,0].plot(self.death_fitters[i].data['cumul'], linestyle = 'dashed')
            hosp = self.axs[i,0].plot(self.hosp_fitters[i].data['cumul'], linestyle = 'dashed')
            fit_d = self.axs[i,0].plot(self.death_fitters[i].index_init, self.death_fitters[i].best_fit_init_cumul(),
                    color = deaths[0].get_color())
            fit_d2 = self.axs[i,0].plot(self.death_fitters[i].index_post_lockdown, self.death_fitters[i].best_fit_post_cumul(),
                             color = deaths[0].get_color())
            fit_h = self.axs[i,0].plot(self.hosp_fitters[i].index_lockdown, self.hosp_fitters[i].best_fit_lock_cumul(),
                    color = hosp[0].get_color())
            fit_h2 = self.axs[i,0].plot(self.hosp_fitters[i].index_post_lockdown,
                             self.hosp_fitters[i].best_fit_post_cumul(), color = hosp[0].get_color())
            self.axs[i,0].set_yscale('log')
            self.axs[i,0].legend((fit_d[0], fit_h[0], fit_h2[0], fit_d2[0]), 
                    [r'$\rho$ = %.2f' % self.death_fitters[i].r, r'$\rho_E$ = %.2f' % self.hosp_fitters[i].rE,
                     r'$\rho_p$ = %.2f' % self.hosp_fitters[i].r_post, r'$\rho_p$ = %.2f' % self.death_fitters[i].r_post], 
                    loc='best')
            
            self.axs[i,1].plot(self.death_fitters[i].data['daily'], linestyle = 'dashed', color = deaths[0].get_color())
            self.axs[i,1].plot(self.hosp_fitters[i].data['daily'], linestyle = 'dashed', color = hosp[0].get_color())
            fit_d = self.axs[i,1].plot(self.death_fitters[i].index_init, self.death_fitters[i].best_fit_init_daily(),
                    color = deaths[0].get_color())
            fit_d2 = self.axs[i,1].plot(self.death_fitters[i].index_post_lockdown,
                             self.death_fitters[i].best_fit_post_daily(), color = deaths[0].get_color())
            fit_h = self.axs[i,1].plot(self.hosp_fitters[i].index_lockdown, self.hosp_fitters[i].best_fit_lock_daily(),
                    color = hosp[0].get_color())
            fit_h2 = self.axs[i,1].plot(self.hosp_fitters[i].index_post_lockdown,
                             self.hosp_fitters[i].best_fit_post_daily(), color = hosp[0].get_color())
            self.axs[i,1].set_yscale('log')
            self.axs[i,1].legend((fit_d[0], fit_h[0], fit_h2[0], fit_d2[0]), 
                    [r'$\rho$ = %.2f' % self.death_fitters[i].r, r'$\rho_E$ = %.2f' % self.hosp_fitters[i].rE,
                     r'$\rho_p$ = %.2f' % self.hosp_fitters[i].r_post, r'$\rho_p$ = %.2f' % self.death_fitters[i].r_post],
                    loc = 'best')
        self.fig.legend((deaths[0], hosp[0]), ['Hospital deaths', 'Hospital admissions'], loc = 'upper right')
        self.axs[self.n-1,0].set_xticks(self.deaths.index[0::tick_interval])
        self.axs[self.n-1,0].set_xticklabels(self.deaths.index[0::tick_interval])
        self.axs[self.n-1,1].set_xticks(self.deaths.index[0::tick_interval])
        self.axs[self.n-1,1].set_xticklabels(self.deaths.index[0::tick_interval])
#        self.fig.set_tight_layout(True)
    
    def plot_fit_lockdown(self):
        tick_interval = 20
        m = int(np.ceil(np.sqrt(self.n)))
        gs = gridspec.GridSpec(m, m)
        fig = plt.figure(dpi = 100, figsize = (12, 8))
        lines = []
        for i in np.arange(self.n):
            x = np.floor_divide(i, m)
            y = np.mod(i, m)
            if i == 0:
                ax = plt.subplot(gs[x, y])
            else:
                ax = plt.subplot(gs[x, y], sharey = ax)
            ax.set_title(self.names[i])
            for fitter in [self.hosp_fitters[i], self.death_fitters[i]]:
                data = ax.plot(fitter.data.index, fitter.data['daily'].values, linestyle = 'dashed')
                ax.plot(fitter.index_lockdown, fitter.best_fit_lock_daily(), 
                        linestyle = 'solid', color = data[0].get_color())
                ax.plot(fitter.index_post_lockdown, fitter.best_fit_post_daily(), 
                        linestyle = 'solid', color = data[0].get_color(),
                        label = r'$\rho_L$ = %.2f, $\rho_E$ = %.2f' % (fitter.rE, fitter.r_post))
                if i == self.n-1:
                    lines.append(data[0])
            ax.legend(loc = 'best')
            ax.set_yscale('log')
            ax.set_xticks(fitter.data.index[0::tick_interval])
            ax.set_xticklabels(fitter.data.index[0::tick_interval])
            ax.grid(True)
        fig.legend(lines, ['Daily hospital admissions', 'Daily hospital deaths'], loc = (.53, .4))
        fig.set_tight_layout(True)
            
    
    def compute_sir(self, p_reported, p_death, end_of_run, Markov = False, verbose = True):
        delay_hosp = delay_hosp_covid()
        delay_death = delay_death_covid()
        p_hosp = np.zeros(self.n)
        for (i, n) in enumerate(self.names):
            p_hosp[i] = p_death*(self.hosp[n].values[1]/self.deaths[n].values[1])*(
                    np.sum(delay_death[:,1]*np.exp(-self.r[i]*delay_death[:,0]))/
                  np.sum(delay_hosp[:,1]*np.exp(-self.r[i]*delay_hosp[:,0])))
        if verbose:
            print('Probabilities of hospitalisation: ', p_hosp)
        EI_dist = EI_dist_covid(p_reported)
        self.sir = []
        time_after_lockdown = (date.datetime.strptime(end_of_run, self.date_format)-
                               date.datetime.strptime(self.lockdown_end_date, self.date_format)).days
        if Markov:
            print('Markov model')
            infectious_time = np.sum(EI_dist[:,1]*EI_dist[:,2])
            incubation_time = np.sum(EI_dist[:,0]*EI_dist[:,2])
            for i in np.arange(self.n):
                self.sir.append(SEIR_lockdown_mixed_delays(self.sizes[i], self.r[i], self.rE[i], 
                                                                    p_death, incubation_time, 
                                                                    infectious_time, delay_death))
        else:
            for i in np.arange(self.n):
                self.sir.append(SEIR_nonMarkov(self.sizes[i], self.r[i], self.rE[i], p_death, 
                                                                    EI_dist, delay_death))
        for (i, sir) in enumerate(self.sir):
            if verbose:
                print('Running SEIR model in ' + self.names[i])
            sir.calibrate(self.deaths_at_lockdown[i], self.lockdown_date)
            if i==1:
                sir.run_two_step_measures(self.date_first_measures_GE, self.r_GE, self.lockdown_length, time_after_lockdown, 
                                          self.r_post[i], verbose = verbose)
            else:
                sir.run_full(self.lockdown_length, time_after_lockdown, self.r_post[i], verbose = verbose)
            sir.compute_deaths()
            sir.compute_hosp(p_hosp[i], delay_hosp)
            time.sleep(.001)
        
#    scale = np.array([1, 1, 1])
    def _fit_beta(self, params):
#        params = self.scale*params
#        params = np.abs(params)
#        params[0] = np.minimum(params[0], 1)
#        params[1:] = 10*params[1:]
#        print(params)
#        print(params)
#        params[0] = .5*(np.tanh(params[0])+1)*(30/24)-20/24
#        params[1] = .5*(np.tanh(params[1])+1)*(20/21 + (8/7)*params[0])-(10/21 + (4/7)*params[0])
#        x = params[0]*np.array([.4, -.8, .4]) + params[1]*np.array([.7, 0, -.7])
#        x = x + np.ones(3)/3
#        x = np.maximum(x, np.zeros(3))
#        x = np.minimum(x, np.ones(3))
#        x = x/np.sum(x)
#        print(x)
#        params[0] = .5*(np.tanh(.01*params[0])+1)*14
#        params[1] = .5*(np.tanh(.01*params[1])+1)*14
#        params[2] = .5*(np.tanh(params[2])+1)*10-5
#        params[1] = 11 + .5*(np.tanh(.1*params[1])+1)*(26-11)
        xE = .5*(np.tanh(params[0])+1)
        xI = .5*(np.tanh(params[1])+1)
        xh = .5*(np.tanh(params[2])+1)
        xd = .5*(np.tanh(params[3])+1)
        vE = 4
        vI = 3
        vh = np.abs(params[4])
        vd = np.abs(params[5])
        
        E_dist = beta_dist(xE*vE, (1-xE)*vE)
        E_dist[:,0] = 2+3*E_dist[:,0]
        I_dist = beta_dist(xI*vI, (1-xI)*vI)
        I_dist[:,0] = 2+10*I_dist[:,0]
        EI_dist = product_dist(E_dist, I_dist)
        print(params)
#        p_hosp = .025
        f = .005
        delay_death = beta_dist(xd*vd, (1-xd)*vd, n = 20)
        delay_death[:,0] = 7 + 20*delay_death[:,0]
        delay_hosp = beta_dist(xh*vh, (1-xh)*vh, n = 20)
        delay_hosp[:,0] = 6 + 10*delay_hosp[:,0]
#        EI_dist = np.array([[3, 2, x[0]], [3, 5, x[1]], [3, 14, x[2]]])
#        delay_hosp = np.array([[params[0], 1]])
#        delay_death = np.array([[params[1], 1]])
#        EI_dist = np.array([[3, 2, .6], [3, 14, .4]])
        self.compute_sir(EI_dist, f, delay_death, delay_hosp)
        return self.error_at('2020-04-20', '2020-04-20')
    
    def _fit_fixed(self, params):
#        params[0] = .5*(np.tanh(params[0])+1)*(30/24)-20/24
#        params[1] = .5*(np.tanh(params[1])+1)*(20/21 + (8/7)*params[0])-(10/21 + (4/7)*params[0])
#        l1 = np.maximum(np.minimum(params[0], 1), 0)*(30/24)-20/24
#        x_ = np.maximum(np.minimum(params[1], 1), 0)
#        l2 = (20/21 + (8/7)*l1)*(2*x_-1)
#        x = l1*np.array([.4, -.8, .4]) + l2*np.array([.7, 0, -.7]) + np.ones(3)/3
#        x = np.maximum(x, np.zeros(3))
#        x = np.minimum(x, np.ones(3))
#        x = x/np.sum(x)
#        print(x)
        p = np.maximum(np.minimum(params, np.ones(self.n)), np.zeros(self.n))
        EI_dist = np.array([[3.5, 3.5, x[0]], [3.5, 6, x[1]], [3.5, 10, x[2]]])
#        xh = .5*(np.tanh(params[2])+1)
#        xd = .5*(np.tanh(params[3])+1)
        xh = np.maximum(np.minimum(params[2], 1), 0)
        xd = np.maximum(np.minimum(params[3], 1), xh/2)
        vh = 5
        vd = 5
        f = .005
        delay_death = beta_dist(xd*vd, (1-xd)*vd, n = 20)
        delay_death[:,0] = 7 + 20*delay_death[:,0]
        delay_hosp = beta_dist(xh*vh, (1-xh)*vh, n = 20)
        delay_hosp[:,0] = 6 + 10*delay_hosp[:,0]
        print('mean delay death', np.sum(delay_death[:,0]*delay_death[:,1]))
        print('mean delay hosp', np.sum(delay_hosp[:,0]*delay_hosp[:,1]))
        self.compute_sir(EI_dist, f, delay_death, delay_hosp)
        return self.error_at('2020-04-20', '2020-04-20')
    
    def _fit_reported(self, params):
        print(params)
        delay_hosp = delay_hosp_covid()
        delay_death = delay_death_covid()
        E_dist = np.array([[3, 1]])
        I_dist = np.concatenate(([1, params[2]]*([3, 0] + [2, 1]*beta_dist(2, 2)),
                  [1, 1-params[2]]*([5, 0] + [10, 1]*beta_dist(2, 2))), axis = 0)
        EI_dist = EI_dist_covid(params[0])
        f = .005
        self.compute_sir(EI_dist, f, delay_death, delay_hosp, end_of_run = '2020-05-11', verbose = True, Markov = False)
        return self.mean_error()
    
    #best fit so far : [38, 36, .85] keep this for now and do another run later ?
    # choose a simple distribution for delay hosp to death
    def fit_mcmc(self, T, init_params):
        beta = 1
        T = int(T)
        params = init_params
        current_fit = self._fit_reported(params)
        self.mcmc_traj = np.zeros((T, np.size(params)))
        self.mcmc_fit = np.zeros(T)
        for i in np.arange(T):
            p = self.propose(params)
            fit = self._fit_reported(p)
            print(np.exp(-beta*(fit-current_fit)))
            if fit < current_fit or np.random.binomial(1, np.exp(-beta*(fit-current_fit))):
                print('accept')
                params = p
                current_fit = fit
            self.mcmc_traj[i,:] = params
            self.mcmc_fit[i] = fit
        k = np.argmin(self.mcmc_fit)
        self.best_fit = self.mcmc_traj[k,:]
        print(self.best_fit)
        
    def propose(self, params):
        k = np.random.choice(np.arange(np.size(params)))
#        print(k)
        if k in [0, 1]:
            params[k] = np.abs(params[k] + np.random.normal(0, 1))
            if k == 0 and params[0] > 25:
                params[0] = np.maximum(0, 2*25-params[0])
        elif k == 2:
            if params[k] - .05 < 0:
                params[k] += .05
            elif params[k] + .05 > 1:
                params[k] -= .05
            else:
                params[k] += (2*np.random.binomial(1, .5)-1)*.05
        return params
    
    def error_at(self, date_death, date_hosp):
        assert hasattr(self, 'sir')
        E = 0
        delay_death = (date.datetime.strptime(date_death, self.date_format)-self.datetime_lockdown).days
        delay_hosp = (date.datetime.strptime(date_hosp, self.date_format)-self.datetime_lockdown).days
        for (i, sir) in enumerate(self.sir):
            x = self.deaths[self.names[i]][date_death]
            y = sir.deaths[sir.shift(sir.lockdown_time + delay_death)]
#            print(x,y)
            E += (1-y/x)**2
            x = self.death_fitters[i].scale*np.prod(self.death_fitters[i].fit_params)
            y = sir.daily_deaths[sir.shift(sir.lockdown_time + delay_death)]
#            print(x,y)
            E += (1-y/x)**2
            x = self.hosp[self.names[i]][date_hosp]
            y = sir.hosp[sir.shift(sir.lockdown_time + delay_hosp)]
#            print(x,y)
            E += (1-y/x)**2
            x = self.hosp_fitters[i].scale*np.prod(self.hosp_fitters[i].fit_params)
            y = sir.daily_hosp[sir.shift(sir.lockdown_time + delay_hosp)]
#            print(x,y)
            E += (1-y/x)**2
        return E
    
    def mean_error(self):
        assert hasattr(self, 'sir')
        E = 0
        for (i, sir) in enumerate(self.sir):
            end_date = (self.datetime_lockdown + date.timedelta(days = sir.times[-1] - sir.lockdown_time))
            delta = (date.datetime.strptime(self.deaths.index[-1], self.date_format)-end_date).days
            end_date = end_date.strftime(self.date_format)
            if not end_date in self.deaths.index:
                end_date = self.deaths.index[-1]
                delta = 0
            x = self.deaths[self.names[i]][:end_date].values
            dt = sir.times_death[1]-sir.times_death[0]
            y = sir.deaths[np.floor((self.observed_times_deaths[:-delta-1]+sir.lockdown_time)/dt).astype(int)]
            E += np.mean((1-y/x)**2)
            x = self.hosp[self.names[i]][:end_date]
            dt = sir.times_hosp[1]-sir.times_hosp[0]
            y = sir.hosp[np.floor((self.observed_times_hosp[:-delta-1]+sir.lockdown_time)/dt).astype(int)]
            E += np.mean((1-y/x)**2)
        return E
    
    def fit_data(self, init_params, bounds = []):
        if bounds == []:
            self.result = optim.minimize(self._fit_fixed, init_params, method = 'Nelder-Mead')
        else:
            self.result = optim.minimize(self._fit_fixed, init_params, method = 'L-BFGS-B', bounds = bounds)
        print(self.result.x)
    
    def plot_deaths_hosp(self, logscale = True):
        self.fig, self.dhaxs = plt.subplots(self.n, 2, dpi = 200, figsize = (10, 16), sharex = True)
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
            if logscale:
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
            self.ax_tot.plot(sir.times_death-sir.lockdown_time, sir.deaths, label = self.names[i])
        self.ax_tot.plot(self.sir[j].times_death-self.sir[j].lockdown_time, deaths, 
                         label = 'predicted deaths in France')
        self.ax_tot.legend(loc='best')
        self.ax_tot.set_yscale('log')
        self.ax_tot.set_xlim((-30,60))
        self.ax_tot.set_ylim((1, 2.5e4))
        self.ax_tot.set_xlabel('Time (days since lockdown)')
        self.ax_tot.set_ylabel('Cumulative hospital deaths')
        self.ax_tot.set_title('Predicted and observed hospital deaths using the 3-patches model')
        self.ax_tot.grid(True)
    
    def compute_deaths_tot(self):
        assert hasattr(self, 'sir')
        shifts = [sir.shift(sir.lockdown_time) for sir in self.sir]
        j = np.argmax(shifts)
        shifts = np.max(shifts)-shifts
        self.deaths_tot = np.zeros(np.max([np.size(sir.times_death) for sir in self.sir]))
        for (i, sir) in enumerate(self.sir):
            self.deaths_tot[shifts[i]:] = self.deaths_tot[shifts[i]:] + sir.deaths
        self.times_deaths_tot = self.sir[j].times_death-self.sir[j].lockdown_time
        
    
    def plot_fit_init(self, deaths_tot, p_reported, p_death):
        self.tot_fitter = Fitter(deaths_tot, self.lockdown_date, self.delay_deaths)
        self.tot_fitter.fit_init('2020-03-01', self.end_fit_init)
        self.observed_times_tot = self.index_to_time(self.tot_fitter.data.index)
#        self.interval_fit = self.index_to_time(self.tot_fitter.index_init)
        
        self.fig, self.init_axs = plt.subplots(1, 2, dpi = 200, figsize = (9, 5))
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
            self.init_axs[1].plot(sir.times_death-sir.lockdown_time, sir.deaths, linewidth = 1.5,
                         color = self.colors[1+i])
        
        self.r_GE = self.r[1]
        self.compute_sir(p_reported, p_death, '2020-05-30')
        self.compute_deaths_tot()
        self.init_axs[0].plot(self.times_deaths_tot, self.deaths_tot, linewidth = 1.5, color = self.colors[0], 
                     label = 'France')
        for (i, sir) in enumerate(self.sir):
            self.init_axs[0].plot(sir.times_death-sir.lockdown_time, sir.deaths, linewidth = 1.5,
                         color = self.colors[1+i], label = self.names[i])
        self.init_axs[0].legend(loc = 'best')
        
        for i in np.arange(2):
            self.init_axs[i].set_ylim((.5, 4e4))
            self.init_axs[i].set_xlim((-40, 40))
            self.init_axs[i].set_xlabel('Time (days since lockdown)')
        self.fig.set_tight_layout(True)
    
    def plot_R0(self):
        n = 100
        p_reported = np.linspace(0, 1, n)
        R0_pre = np.zeros((n, self.n))
        R0_lock = np.zeros((n, self.n))
        for i in np.arange(n):
            EI_dist = EI_dist_covid(p_reported[i])
            for j in np.arange(self.n):
                R0_pre[i, j] = R0(self.r[j], EI_dist)
                R0_lock[i, j] = R0(self.rE[j], EI_dist)
        plt.figure(dpi = 100, figsize = (6, 6))
        self.axs = plt.axes()
        for i in np.arange(self.n):
            self.axs.plot(p_reported, R0_pre[:,i], label = '$R_0$ before lockdown in ' + self.names[i])
            self.axs.plot(p_reported, R0_lock[:,i], label = '$R_0$ during lockdown in ' + self.names[i])
        self.axs.legend(loc = 'best')
        self.axs.set_yscale('log')
        self.axs.set_xlabel('Proportion of reported individuals')
    
    linestyles = ['solid', 'dashed', 'dashdot']
    
    def plot_immunity(self, f_values, p_reported, logscale = False):
        self.fig, self.axs = plt.subplots(1, self.n, dpi = 200, figsize = (12, 4), sharey = True, sharex = True)
        for i in np.arange(self.n):
            self.axs[i].set_title(self.names[i])
            self.axs[i].set_xlabel('Time (days since lockdown)')
            self.axs[i].grid(True)
            self.axs[i].yaxis.set_tick_params(which = 'both', labelleft = True)
            if logscale:
                self.axs[i].set_yscale('log')
        self.axs[0].set_ylabel('Proportion of infected individuals (1-S)', fontsize = 11)
        
        for (j, f) in enumerate(f_values):
            self.compute_sir(p_reported, f, '2020-06-15')
            for (i, sir) in enumerate(self.sir):
                self.axs[i].plot(sir.times-sir.lockdown_time, 1-sir.traj[:,0],
                        label = 'f = %.2f%%' % (100*f), 
                        linestyle = self.linestyles[np.mod(j, np.size(self.linestyles))])
        self.axs[2].legend(loc = 'upper right', title = 'Infection fatality ratio')
        self.fig.set_tight_layout(True)
    
    def plot_SIR_deaths_hosp(self):
        assert hasattr(self, 'sir')
        self.fig, self.faxs = plt.subplots(self.n, 3, dpi = 200, figsize = (12, 12), sharex = False)
        self.faxs[0, 0].set_title('trajectory of the SEIR model')
        self.faxs[0, 1].set_title('Hospital admissions')
        self.faxs[0, 2].set_title('Hospital deaths')
        for i in np.arange(self.n):
            self.faxs[self.n-1, i].set_xlabel('Time (days since lockdown)')
        for (i, sir) in enumerate(self.sir):
            self.faxs[i,0].set_ylabel(self.names[i])
            self.faxs[i,0].plot(sir.times-sir.lockdown_time, 1-sir.traj[:,0], label = '1-S')
            self.faxs[i,0].plot(sir.times-sir.lockdown_time, sir.traj[:,3], label = 'E')
            self.faxs[i,0].plot(sir.times-sir.lockdown_time, sir.traj[:,1], label = 'I')
            self.faxs[i,0].plot(sir.times-sir.lockdown_time, sir.traj[:,2], label = 'R')
            self.faxs[i,0].grid(True)
            self.faxs[i,0].vlines(0, 0, .1, linestyle = 'dashed')
            self.faxs[i,0].legend(loc='best')
            self.faxs[i,0].set_ylim((0, .1))
            
            cumul = self.faxs[i,1].plot(sir.times_hosp-sir.lockdown_time, sir.hosp, label = 'Cumulative')
            daily = self.faxs[i,1].plot(sir.times_daily_hosp-sir.lockdown_time, sir.daily_hosp, label = 'Daily')
            self.faxs[i,1].plot(self.observed_times_hosp, self.hosp[self.names[i]].values, linestyle = 'dashed',
                     color = cumul[0].get_color())
            self.faxs[i,1].plot(self.observed_times_hosp[1:], np.diff(self.hosp[self.names[i]].values),
                     linestyle = 'dashed', color = daily[0].get_color())
            self.faxs[i,1].set_yscale('log')
            self.faxs[i,1].set_ylim((1e0, 6e4))
            self.faxs[i,1].grid(True)
            self.faxs[i,1].legend(loc='best')
            self.faxs[i,1].set_xlim((-7, np.max(self.observed_times_hosp)))
            
            cumul = self.faxs[i,2].plot(sir.times_death-sir.lockdown_time, sir.deaths, label = 'Cumulative')
            daily = self.faxs[i,2].plot(sir.times_daily_deaths-sir.lockdown_time, sir.daily_deaths, label = 'Daily')
            self.faxs[i,2].plot(self.observed_times_deaths, self.deaths[self.names[i]].values,
                     linestyle = 'dashed', color = cumul[0].get_color())
            self.faxs[i,2].plot(self.observed_times_deaths[1:], np.diff(self.deaths[self.names[i]].values), 
                     linestyle = 'dashed', color = daily[0].get_color())
            self.faxs[i,2].set_yscale('log')
            self.faxs[i,2].set_ylim((1e-1, 1.6e4))
            self.faxs[i,2].grid(True)
            self.faxs[i,2].legend(loc = 'best')
            self.faxs[i,2].set_xlim((-7, np.max(self.observed_times_deaths)))
        self.fig.set_tight_layout(True)
    
    def plot_markov_vs_nonmarkov(self, p_reported, p_death):
        self.fig, self.axs = plt.subplots(self.n, 3, dpi = 200, figsize = (11, 9))
        self.axs[0,0].set_title('Proportion of infected individuals (1-S)')
        self.axs[0,1].set_title('Daily hospital admissions')
        self.axs[0,2].set_title('Daily hospital deaths')
        for i in np.arange(3):
            self.axs[self.n-1, i].set_xlabel('Time (days since lockdown)')
        for i in np.arange(self.n):
            self.axs[i,0].set_ylabel(self.names[i], fontsize = 12)
            self.axs[i,0].set_ylim((-.01, .09))
            for j in np.arange(3):
                self.axs[i,j].grid(True)
            
        # plot data
        for (k, fit_list) in enumerate([self.hosp_fitters, self.death_fitters]):
            for (i, fitter) in enumerate(fit_list):
                self.axs[i,1+k].plot(self.observed_times_deaths, fitter.data['daily'].values,
                        linestyle = 'dashed', color = self.colors[k], linewidth = 1.2)
        
        self.compute_sir(p_reported, p_death, self.deaths.index[-1])
        # plot non-Markov
        for (i, sir) in enumerate(self.sir):
            self.axs[i,0].plot(sir.times-sir.lockdown_time, 1-sir.traj[:,0], 
                    color = self.colors[2], label = 'non-Markovian SEIR model')
            self.axs[i,1].plot(sir.times_daily_hosp-sir.lockdown_time, sir.daily_hosp, 
                    color = self.colors[2], linewidth = 1.5)
            self.axs[i,2].plot(sir.times_daily_deaths-sir.lockdown_time, sir.daily_deaths, 
                    color = self.colors[2], linewidth = 1.5)
        
        self.compute_sir(p_reported, p_death, self.deaths.index[-1], Markov = True)
        # plot Markov
        for (i, sir) in enumerate(self.sir):
            self.axs[i,0].plot(sir.times-sir.lockdown_time, 1-sir.traj[:,0],
                    linestyle = 'dashdot', color = self.colors[3], label = 'Markovian SEIR model')
            self.axs[i,1].plot(sir.times_daily_hosp-sir.lockdown_time, sir.daily_hosp,
                    linestyle = 'dashdot', color = self.colors[3], linewidth = 1.5)
            self.axs[i,2].plot(sir.times_daily_deaths-sir.lockdown_time, sir.daily_deaths,
                    linestyle = 'dashdot', color = self.colors[3], linewidth = 1.5)
        self.axs[self.n-1,0].legend(loc = 'best')
            
    
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


