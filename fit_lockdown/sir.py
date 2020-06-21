#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:44:32 2020

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as date

from fit_lockdown import *

class SIR(object):
#    e = np.array([-1, 1])
    def _setZ(self, Z):
#        print(Z)
#        assert np.size(Z) == self.n and np.abs(np.sum(Z)-1) < 1e-8 and np.min(Z) >= 0
        assert np.size(Z) == self.n and np.min(Z) >= 0
        self._Z = np.log(Z)
    def _getZ(self):
        return np.exp(self._Z)
    Z = property(_getZ, _setZ)
    
    def _setl(self, l):
        assert l > 0
        self._l = l
    def _getl(self):
        return self._l
    l = property(_getl, _setl)
    
    def _setmu(self, mu):
        assert mu > 0
        self._mu = mu
    def _getmu(self):
        return self._mu
    mu = property(_getmu, _setmu)
    
    n = 3
    
    def __init__(self, contact_rate, remission_rate):
#        print('SIR.__init__')
        self.l = contact_rate
        self.mu = remission_rate
        self.Z = np.concatenate(([1], np.zeros(self.n-1)))
    
    def run(self, T, dt = .005, record= False):
        assert dt > 0
        n = int(T/dt)
        if record and hasattr(self, 'traj'):
            last_t = self.times[-1]
            offset = np.size(self.times)-1
            self.times = np.concatenate((self.times, last_t + dt*np.arange(1,n+1)))
            self.traj = np.concatenate((self.traj, np.zeros((n,self.n))), axis = 0)
        elif record:
            self.times = dt*np.arange(n+1)
            self.traj = np.zeros((n+1,self.n))
            self.traj[0,:] = self.Z
            offset = 0
        for t in np.arange(n):
            self._step(dt)
            if record:
                self.traj[offset+t+1,:] = self.Z
    
    def forget(self):
        if hasattr(self, 'traj'):
            del self.traj
        if hasattr(self, 'times'):
            del self.times
        
    def _step(self, dt):
        S = self._Z[0] - self.l*dt*self.Z[1]
        I = self._Z[1] + dt*(self.l*self.Z[0] - self.mu)
        R = np.log(1-np.exp(S)-np.exp(I))
        self._Z = [S, I, R]
    
    def plot(self, S = True):
        assert hasattr(self, 'traj')
        plt.figure(dpi=200)
        self.ax = plt.axes()
        if S:
            self.ax.plot(self.times, self.traj[:,0], label = 'S', linestyle = 'dashed')
        self.ax.plot(self.times, self.traj[:,1], label = 'I', linestyle = 'solid')
        self.ax.plot(self.times, self.traj[:,2], label = 'R', linestyle = 'dashdot')
        self.ax.legend(loc='best')
        self.ax.set_xlabel('time (days)')
    
    def R0(self):
        return self.l/self.mu

class SEIR(SIR):
    def _set_nu(self, nu):
        assert nu > 0
        self._nu = nu
    def _get_nu(self):
        return self._nu
    nu = property(_get_nu, _set_nu)
    
    n = 4
    
    def __init__(self, contact_rate, remission_rate, symptom_rate):
#        print('SEIR.__init__')
        self.nu = symptom_rate
        if not hasattr(self, 'l'):
            SIR.__init__(self, contact_rate, remission_rate)
    
    def _step(self, dt):
        S = self.Z[0]*np.exp(-self.l*dt*self.Z[1])
        R = self.Z[2] + self.mu*dt*self.Z[1]
        E = self.Z[3] - (S-self.Z[0]) - self.nu*dt*self.Z[3]
        I = 1-np.sum([S, R, E])
        self.Z = [S, I, R, E]
    
    def plot(self, S = True):
        super().plot(S)
        self.ax.plot(self.times, self.traj[:,3], label = 'E', linestyle = (0, (3, 1, 1, 1, 1, 1)))
        self.ax.legend(loc = 'best')

class SIR_lockdown(SIR):
    date_format = '%Y-%m-%d'
    
    def __init__(self, N, growth_rate_init, growth_rate_lockdown, case_fatality_rate, generation_time, delay):
#        print('SIR_lockdown.__init__')
        assert N > 0
        self.N = float(N)
        assert case_fatality_rate > 0 and generation_time > 0 and delay > 0
        self.g = generation_time
        assert self.contact_rate(growth_rate_init) > 0
        assert self.contact_rate(growth_rate_lockdown) > 0
        self.r = growth_rate_init
        self.rE = growth_rate_lockdown
        self.f = case_fatality_rate
        self.delay = delay
        if not hasattr(self, 'l'):
            SIR.__init__(self, self.contact_rate(self.r), self.g**-1)
    
    def contact_rate(self, r):
        return r + self.g**-1
    
    def contact_rate_R0(self, R0):
        assert R0 > 0
        return R0*self.g**-1
    
    def calibrate(self, deaths_lockdown, lockdown_date, I0 = 1):
        assert I0 > 0 and deaths_lockdown > 0
        self.lockdown_date = date.datetime.strptime(lockdown_date, self.date_format)
        self.lockdown_time = self.delay + np.log(deaths_lockdown/(self.f*I0))/self.r
        self.Z = np.concatenate(([1-I0/self.N], I0/self.N*self.lead_eigen_vect()))
    
    def lead_eigen_vect(self):
        return np.array([self.r/self.l, 1-self.r/self.l])
    
    def run_full(self, lockdown_length, time_after_lockdown, r_after_lockdown, verbose = True):
        assert lockdown_length > 0 and time_after_lockdown >= 0
        assert hasattr(self, 'lockdown_time')
        if verbose:
            print('R0 prior to lockdown: %.2f' % self.R0())
        self.run(self.lockdown_time, record = True)
#        print('State at the start of lockdown: ', self.Z)
#        self.l = self.contact_rate(self.rE)/self.Z[0]
        self.l = self.contact_rate(self.rE)
        if verbose:
            print('R0 during lockdown: %.2f' % self.R0())
        self.run(lockdown_length, record = True)
        if verbose:
            print('State at the end of lockdown: ', self.Z)
        self.l = self.contact_rate(r_after_lockdown)/self.Z[0]
        self.run(time_after_lockdown, record = True)
#        print('Final state: ', self.Z)
        self.lockdown_length = lockdown_length
    
    def run_two_step_measures(self, date_of_measures, growth_rate_before_measures, 
                              lockdown_length, time_after_lockdown, r_after_lockdown, verbose = True):
        assert lockdown_length > 0 and time_after_lockdown >= 0
        assert hasattr(self, 'lockdown_time')
        assert growth_rate_before_measures > 0
        self.date_measures = date.datetime.strptime(date_of_measures, self.date_format)
        tau = (self.lockdown_date-self.date_measures).days
        x = (self.r/growth_rate_before_measures)*(self.lockdown_time-tau)
        self.lockdown_time = x + tau
        self.l = self.contact_rate(growth_rate_before_measures)
        if verbose:
            print("R0 before first measures: %.2f" % self.R0())
        self.run(x, record = True)
        self.l = self.contact_rate(self.r)
        if verbose:
            print("R0 after first measures: %.2f" % self.R0())
        self.run(tau, record = True)
        self.l = self.contact_rate(self.rE)
        if verbose:
            print("R0 during lockdown: %.2f" % self.R0())
        self.run(lockdown_length, record = True)
        if verbose:
            print("State at the end of lockdown: ", self.Z)
        self.l = self.contact_rate(r_after_lockdown)/self.Z[0]
        self.run(time_after_lockdown, record = True)
        self.lockdown_length = lockdown_length
    
    def plot(self, S = True):
        super().plot(S)
        if S:
            ymax = 1
        else:
            ymax = np.max(self.traj[:,1:])
        if hasattr(self, 'lockdown_length'):
            self.ax.vlines(self.lockdown_time, 0, ymax, label = 'lockdown')
            if self.times[-1] > self.lockdown_time + self.lockdown_length:
                self.ax.vlines(self.lockdown_time + self.lockdown_length, 0, ymax, 
                               label = 'easing of lockdown, $R_0$ = %.1f' % self.R0(), linestyle = 'dotted')
            self.ax.legend(loc='best')
    
    def compute_deaths(self):
        assert hasattr(self, 'traj')
        self.times_death = self.times+self.delay
        self.deaths = self.f*self.N*(1-self.traj[:,0])
        self.daily_deaths = np.concatenate(([0], np.diff(self.deaths)/np.diff(self.times_death)))

class SEIR_lockdown(SIR_lockdown, SEIR):
    def __init__(self, N, growth_rate_init, growth_rate_lockdown, case_fatality_rate, generation_time, delay, incubation_time):
#        print('SEIR_lockdown.__init__')
        assert incubation_time > 0
        self.incubation_time = incubation_time
        if not hasattr(self, 'N'):
            SIR_lockdown.__init__(self, N, growth_rate_init, growth_rate_lockdown, case_fatality_rate, generation_time, delay)
        SEIR.__init__(self, self.l, self.mu, self.incubation_time**-1)
    
    def contact_rate(self, r):
        return self.g**-1 + (r*self.incubation_time)*(self.incubation_time**-1 + self.g**-1 + r)
    
    def lead_eigen_vect(self):
        x = self.nu/(self.nu+self.r)
        y = self.mu/(self.mu+self.r)
        e = 1-x
        i = x*(1-y)
        r = x*y
        return np.array([i, r, e])

class SIR_lockdown_mixed_delays(SIR_lockdown):
    def __init__(self, N, growth_rate_init, growth_rate_lockdown, case_fatality_rate, 
                 generation_time, delay_dist):
#        print('SIR_lockdown_mixed_delays.__init__')
        if not hasattr(self, 'N'):
            SIR_lockdown.__init__(self, N, growth_rate_init, growth_rate_lockdown, case_fatality_rate, 
                              generation_time, 1)
        assert is_dist(delay_dist)
        self.delay_dist = delay_dist
     
    def calibrate(self, deaths_lockdown, lockdown_date, I0 = 1):
        super().calibrate(deaths_lockdown, lockdown_date, I0)
        denom = np.sum(self.delay_dist[:,1]*np.exp(-self.r*self.delay_dist[:,0]))
        self.lockdown_time = np.log(deaths_lockdown/(self.f*I0*denom))/self.r
    
    def shift(self, delay):
        return np.argmax(self.times > delay)-1
    
    def compute_deaths(self):
        assert hasattr(self, 'traj')
        self.times_death = self.times
        self.deaths = np.zeros(np.size(self.times))
        for a in self.delay_dist:
            i = self.shift(a[0])
            time_range = np.arange(i,np.size(self.times))
            self.deaths[time_range] += self.f*self.N*a[1]*(1-self.traj[time_range-i,0])
        day = self.shift(1)
        self.times_daily_deaths = self.times_death[day:]
        self.daily_deaths = self.deaths[day:]-self.deaths[:-day]
    
    def compute_hosp(self, p_hosp, delay_hosp):
        assert is_dist(delay_hosp) and p_hosp > 0 and p_hosp <= 1
        record = False
        if hasattr(self, 'deaths'):
            record = True
            _deaths = self.deaths
            _death_times = self.times_death
            _times_daily_deaths = self.times_daily_deaths
            _daily_deaths = self.daily_deaths
        _f = self.f
        _delay_death = self.delay_dist
        self.f = p_hosp
        self.delay_dist = delay_hosp
        self.compute_deaths()
        self.times_hosp = self.times_death
        self.hosp = self.deaths
        self.times_daily_hosp = self.times_daily_deaths
        self.daily_hosp = self.daily_deaths
        self.f = _f
        self.delay_dist = _delay_death
        if record:
            self.deaths = _deaths
            self.daily_deaths = _daily_deaths
            self.times_daily_deaths = _times_daily_deaths
            self.times_death = _death_times
    
    def plot_deaths_fit(self, data):
        assert hasattr(self, 'deaths')
        self.fig, self.dfit_axs = plt.subplots(1,2,dpi=200,figsize=(11,4))
        data_start_date = date.datetime.strptime(data.index[0], self.date_format)
        data_start_time = self.lockdown_time + (data_start_date - self.lockdown_date).days
        observed_interval = data_start_time + np.arange(np.size(data.index))
        self.dfit_axs[0].set_title('Cumulative deaths')
        self.dfit_axs[0].plot(self.times_death, self.deaths, label = 'predicted deaths')
        self.dfit_axs[0].plot(observed_interval, data['cumul'].values, label = 'observed deaths', linestyle = 'dashdot')
        self.dfit_axs[0].vlines(self.lockdown_time, 0, np.max(self.deaths), label = 'Start of lockdown')
        self.dfit_axs[0].legend(loc='best')
        self.dfit_axs[0].set_xlabel('Time (days)')
        self.dfit_axs[1].set_title('Daily deaths')
        self.dfit_axs[1].set_xlabel('Time (days)')
        self.dfit_axs[1].plot(self.times_daily_deaths, self.daily_deaths, label = 'predicted deaths')
        self.dfit_axs[1].plot(observed_interval, data['daily'].values, label = 'observed deaths', linestyle = 'dashdot')
        self.dfit_axs[1].vlines(self.lockdown_time, 0, np.max(self.daily_deaths), label = 'Start of lockdown')
#        self.dfit_axs[1].legend(loc='best')
#        self.dfit_axs[1].set_yscale('log')

class SEIR_lockdown_mixed_delays(SIR_lockdown_mixed_delays, SEIR_lockdown):
    def __init__(self, N, growth_rate_init, growth_rate_lockdown, case_fatality_rate, 
                 generation_time, incubation_time, delay_dist):
#        print('SEIR_lockdown_mixed_delays.__init__')
        SEIR_lockdown.__init__(self, N, growth_rate_init, growth_rate_lockdown, case_fatality_rate, 
                               generation_time, 1, incubation_time)
        if not hasattr(self, 'delay_dist'):
            SIR_lockdown_mixed_delays.__init__(self, N, growth_rate_init, growth_rate_lockdown, 
                                               case_fatality_rate, generation_time, delay_dist)

class SIR_nonMarkov(SIR_lockdown_mixed_delays):
    def __init__(self, N, growth_rate_init, growth_rate_lockdown, case_fatality_rate,
                 infectious_period_dist, delay_dist):
#        print('SIR_nonMarkov.__init__')
        assert is_dist(infectious_period_dist)
        self.I_dist = infectious_period_dist
        if not hasattr(self, 'delay_dist'):
            SIR_lockdown_mixed_delays.__init__(self, N, growth_rate_init, growth_rate_lockdown, 
                                           case_fatality_rate, 1, delay_dist)
        
    def LaplaceI(self, r):
        return np.sum(self.I_dist[:,1]*np.exp(-r*self.I_dist[:,0]))
    
    def EI(self):
        return np.sum(self.I_dist[:,1]*self.I_dist[:,0])
    
    def contact_rate(self, r):
        return r/(1-self.LaplaceI(r))
    
    def contact_rate_R0(self, R0):
        assert R0 > 0
        return R0/self.EI()
    
    def R0(self):
        return self.l*self.EI()
    
    def run(self, T, dt = .01, record = True):
        n = int(T/dt)
        if hasattr(self, 'times'):
            assert dt == self.times[1]-self.times[0]
            self.flux = np.concatenate((self.flux, np.zeros((n, self.n-1))))
        else:
            self.i = 0
            self._init_flux(n, dt)
        super().run(T, dt, True)
        
    def _init_flux(self, n, dt):
        self.flux = np.zeros((n + int(np.max(self.I_dist[:,0])/dt), self.n-1))
        for (I, p) in self.I_dist:
            i = int(I/dt)
            self.flux[0:i, 1] += self.Z[1]*p*self.l*np.exp(self.r*(dt*np.arange(i)-I))*dt
        self.A = np.array([[-1, 0], [1, -1], [0, 1]])
    
    def _step(self, dt):
        ds = self.Z[0]*(1 - np.exp(-self.l*dt*self.Z[1]/np.sum(self.Z)))
        self.flux[self.i,0] = ds
        for (I, p) in self.I_dist:
            self.flux[self.i+int(I/dt),1] += p*ds
        self.Z += np.matmul(self.A, self.flux[self.i,:])
        self.i += 1
    
    def forget(self):
        SIR_lockdown_mixed_delays.forget(self)
        if hasattr(self, 'flux'):
            del self.flux
    
    def check(self):
        assert hasattr(self, 'traj')
        dt = np.concatenate((np.diff(self.times), [0]))
        I1 = np.cumsum(dt*self.l*self.traj[:,0]*self.traj[:,1])
        self.D1 = self.traj[:,0] - self.traj[0,0] + I1
        print(np.max(np.abs(self.D1)))
        F0c = 0*self.times
        I2 = 0*self.times
        for [d,p] in self.I_dist:
            F0c += (self.l/self.r)*p*((self.times >= d) + (self.times < d)*np.exp(self.r*(self.times-d)))
            for (i, t) in enumerate(self.times):
                if t < d:
                    pass
                I2[i] += self.l*p*np.sum(dt*self.traj[:,0]*self.traj[:,1]*(self.times <= t-d))
        self.D2 = self.traj[:,1] - self.traj[0,1]*F0c - I1 + I2
        print(np.max(np.abs(self.D2)))
    
class SEIR_nonMarkov(SIR_nonMarkov, SEIR_lockdown_mixed_delays):
    def __init__(self, N, growth_rate_init, growth_rate_lockdown, case_fatality_rate, 
                 EI_period_dist, delay_dist):
#        print('SEIR_nonMarkov.__init__')
        assert is_dist(EI_period_dist, dim = 2)
        self.EI_dist = EI_period_dist
        self.E_dist = self.EI_dist[:,0::2]
        SIR_nonMarkov.__init__(self, N, growth_rate_init, growth_rate_lockdown, case_fatality_rate, 
                               self.EI_dist[:,1:], delay_dist)
        # I think I do not need to call SEIR_lockdown_mixed_delays.__init__()
        
    def LaplaceE(self, r):
        return np.sum(self.E_dist[:,1]*np.exp(-r*self.E_dist[:,0]))
    
    def LaplaceEI(self, r):
        return np.sum(self.EI_dist[:,2]*np.exp(-r*(self.EI_dist[:,0]+self.EI_dist[:,1])))
    
    def EE(self):
        return np.sum(self.E_dist[:,1]*self.E_dist[:,0])
        
    def contact_rate(self, r):
        return r/(self.LaplaceE(r)-self.LaplaceEI(r))
    
    def lead_eigen_vect(self):
        i = self.r/self.l
        e = 1-self.LaplaceE(self.r)
        r = self.LaplaceEI(self.r)
        return np.array([i, r, e])
    
    def _init_flux(self, n, dt):
        self.flux = np.zeros((n + int(np.max(self.EI_dist[:,0]+self.EI_dist[:,1])/dt), self.n-1))
        self.A = np.array([[-1, 0, 0], [0, 1, -1], [0, 0, 1], [1, -1, 0]])
        g0 = self.r/(1-self.LaplaceE(self.r))
        d0 = self.r/self.LaplaceEI(self.r)
        for (E, I, p) in self.EI_dist:
            e = int(E/dt)
            i = int(I/dt)
            self.flux[0:e,1] += self.Z[3]*g0*p*np.exp(self.r*(np.arange(e)*dt-E))*dt
            self.flux[0:(e+i),2] += self.Z[2]*d0*p*np.exp(self.r*(np.arange(e+i)*dt-(E+I)))*dt
    
    def _step(self, dt):
        ds = self.Z[0]*(1-np.exp(-self.l*dt*self.Z[1]/np.sum(self.Z)))
        self.flux[self.i,0] = ds
        for (E, I, p) in self.EI_dist:
            self.flux[self.i+int(E/dt),1] += p*ds
            self.flux[self.i+int((E+I)/dt),2] += p*ds
        self.Z += np.matmul(self.A, self.flux[self.i,:])
        self.i += 1



#N = 12e6
#r = .3
#rE = -.06
#f = .006
#delays = np.array([[17, 1]])
#g = 10
#I = np.array([[g, 1]])
##sir = SIR_nonMarkov(N, r, rE, f, I, delays)
#sir = SIR_lockdown_mixed_delays(N, r, rE, f, g, delays)
#sir.calibrate(5e2)
#R0 = sir.R0()
##sir.run(100)
#sir.run_full(50, 200, R0)
#sir.plot()

#R01 = 3.2
#R02 = .5
#R03 = 1.8
#N_idf = 12.21e6
#N_france = 67e6
#sir = SIR(R01*.1, .1)
#i0 = 1./N_idf
#sir.Z = [1-i0, i0, 0]
#sir.run(48, record = True)
#sir.l = R02*sir.mu
#sir.run(55, record = True)
#sir.l = R03*sir.mu
#sir.run(200, record = True)
#sir.plot()
#sir.ax.set_title('Predicted epidemic in Ile de France without lockdown, $R_0$=%.1f' % R0)


