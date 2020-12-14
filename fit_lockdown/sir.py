#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:44:32 2020

@author: raphael
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as date
import pandas as pd
import scipy.integrate as integrate

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
    
    def plot(self, axes = None, S = True):
        assert hasattr(self, 'traj')
        if axes == None:
            plt.figure(dpi=200)
            self.ax = plt.axes()
        else:
            self.ax = axes
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
    
    def plot(self, axes = None, S = True):
        super().plot(axes, S)
        self.ax.plot(self.times, self.traj[:,3], label = 'E', linestyle = (0, (3, 1, 1, 1, 1, 1)))
        self.ax.legend(loc = 'best')

class SIR_lockdown(SIR):
    date_format = '%Y-%m-%d'
    
    def __init__(self, growth_rate_init, infectious_time, date_lockdown, pop_size, two_step_measures = False, 
                           growth_rate_before_measures = 0, date_of_measures = ''):
#        print('SIR_lockdown.__init__')
        assert infectious_time > 0
        self.g = infectious_time
        assert pop_size > 0
        self.N = float(pop_size)
        assert self.contact_rate(growth_rate_init) > 0
        self.r = growth_rate_init
        self.lockdown_date = date.datetime.strptime(date_lockdown, self.date_format)
        if not hasattr(self, 'l'):
            SIR.__init__(self, self.contact_rate(self.r), self.g**-1)
        self.two_step_measures = bool(two_step_measures)
        if self.two_step_measures:
            assert growth_rate_before_measures > 0
            self.r_before_measures = growth_rate_before_measures
            self.date_measures = date.datetime.strptime(date_of_measures, self.date_format)
    
    def contact_rate(self, r):
        return r + self.g**-1
    
    def contact_rate_R0(self, R0):
        assert R0 > 0
        return R0*self.g**-1
    
    def R0(self):
        return self.l*self.g
    
    def R0_eff(self):
        return self.R0()*self.Z[0]
    
    def change_contact_rate(self, growth_rate, adjust_S = True, verbose = False):
        if adjust_S:
            z = self.Z[0]
        else:
            z = 1
        self.l = self.contact_rate(growth_rate)/z
        if verbose:
            print('R_0 = %.3f, R0_eff = %.3f' % (self.R0(), self.R0_eff()))
    
    def calibrate(self, deaths_lockdown, infection_fatality_ratio, delay_dist, I0 = 1):
        assert I0 > 0 and deaths_lockdown > 0
        assert is_dist(delay_dist) and np.min(delay_dist[:,0] >= 0)
        assert infection_fatality_ratio > 0 and infection_fatality_ratio <= 1
        denom = Laplace(delay_dist, -self.r)
        self.lockdown_time = np.log(deaths_lockdown/(infection_fatality_ratio*I0*denom))/self.r
        self.change_contact_rate(self.r, adjust_S = False)
        if self.two_step_measures:
            tau = (self.lockdown_date-self.date_measures).days
            num = np.exp(self.r*self.lockdown_time)*denom
            denom1 = np.sum(np.exp(self.r*(tau-delay_dist[:,0]))*(
                    delay_dist[:,0] <= tau)*delay_dist[:,1])
            denom2 = np.sum(np.exp(self.r_before_measures*(tau-delay_dist[:,0]))*(
                    delay_dist[:,0] > tau)*delay_dist[:,1])
            S = num / (denom1 + denom2)
            delta = self.lockdown_time - tau - np.log(S)/self.r_before_measures
#            print(S, delta, tau)
            self.lockdown_time -= delta
            self.init_phase = self.lockdown_time - tau
            self.change_contact_rate(self.r_before_measures, adjust_S = False)
        self.Z = np.concatenate(([1-I0/self.N], I0/self.N*self.lead_eigen_vect()))
    
    def lead_eigen_vect(self, rho = None):
        if rho == None:
            rho = self.r
        l = self.contact_rate(rho)
        return np.array([rho/l, 1-rho/l])
    
    def shift(self, delay):
        return np.argmax(self.times > delay)-1
    
    def lockdown_constants(self, growth_rate_lockdown, delta):
        self.calibrate(1, .1, np.array([[0, 1]]))
        self.run_up_to_lockdown(verbose = False)
        Z_lockdown = self.Z
        # print('S at lockdown while fitting delays: ', self.Z[0])
        self.change_contact_rate(growth_rate_lockdown, adjust_S = False)
        self.run(delta, record = True)
#        self.plot(S = False)
        self.forget()
        return self.Z[1:]/Z_lockdown[1:]
    
    def run_up_to_lockdown(self, verbose = True):
        assert hasattr(self, 'lockdown_time')
        self.forget()
        if not self.two_step_measures:
            if verbose:
                print('R0 prior to lockdown: %.2f' % self.R0())
            self.run(self.lockdown_time, record = True)
        else:
            if verbose:
                print("R0 before first measures: %.2f" % self.R0())
            self.run(self.init_phase, record = True)
            self.change_contact_rate(self.r, adjust_S = False)
            if verbose:
                print("R0 after first measures: %.2f" % self.R0())
            self.run(self.lockdown_time-self.init_phase, record = True)
    
    def plot(self, axes = None, S = True):
        super().plot(axes, S)
        if S:
            ymax = 1
        else:
            ymax = np.max(self.traj[:,1:])
        if hasattr(self, 'lockdown_time'):
            self.ax.vlines(self.lockdown_time, 0, ymax, label = 'lockdown')
            self.ax.legend(loc='best')
    
    def compute_delayed_event(self, ratio, delay_dist, name):
        assert hasattr(self, 'traj')
        assert ratio > 0 and ratio <= 1
        assert is_dist(delay_dist) and np.min(delay_dist[:,0]) >= 0
        if not hasattr(self, 'cumul'):
            self.cumul = pd.DataFrame(index = self.times)
        if not hasattr(self, 'daily'):
            self.daily = pd.DataFrame(index = self.times)
        self.cumul[name] = pd.Series(np.zeros(np.size(self.times)))
        for (delay, p) in delay_dist:
            if delay >= np.max(self.times):
                continue
            i = self.shift(delay)
            time_range = np.arange(i,np.size(self.times))
            self.cumul[name].values[time_range] += ratio*self.N*p*(1-self.traj[time_range-i,0])
        self.daily[name] = np.concatenate(([0], np.diff(self.cumul[name].values)/np.diff(self.times)))
    
    def forget(self):
        super().forget()
        if hasattr(self, 'cumul'):
            del self.cumul
        if hasattr(self, 'daily'):
            del self.daily
    
    def plot_event(self, name, daily = False, axes = None, labels = True, color = None,
                   linewidth = None):
        assert name in self.cumul.columns
        if axes == None:
            plt.figure(dpi = 200)
            self.axes = plt.axes()
        else:
            self.axes = axes
        if daily:
            line, = self.axes.plot(self.daily.index-self.lockdown_time, 
                                   self.daily[name].values)
        else:
            line, = self.axes.plot(self.cumul.index-self.lockdown_time, 
                                   self.cumul[name].values)
        if labels:
            line.set_label(name)
            self.axes.legend(loc = 'best')
        if not color == None:
            line.set_color(color)
        if not linewidth == None:
            line.set_linewidth(linewidth)
        self.axes.grid(True)
        return line
                

class SEIR_lockdown(SIR_lockdown, SEIR):
    def __init__(self, growth_rate_init, infectious_time, incubation_time, date_lockdown, pop_size,
                 two_step_measures = False, growth_rate_before_measures = 0, date_of_measures = ''):
#        print('SEIR_lockdown.__init__')
        assert incubation_time > 0
        self.incubation_time = incubation_time
        if not hasattr(self, 'N'):
            SIR_lockdown.__init__(self, growth_rate_init, infectious_time, date_lockdown, pop_size,
                                  two_step_measures, growth_rate_before_measures, date_of_measures)
        SEIR.__init__(self, self.l, self.mu, self.incubation_time**-1)
    
    def contact_rate(self, r):
        return self.g**-1 + (r*self.incubation_time)*(self.incubation_time**-1 + self.g**-1 + r)
    
    def lead_eigen_vect(self, rho = None):
        if rho == None:
            rho = self.r
        x = self.nu/(self.nu+rho)
        y = self.mu/(self.mu+rho)
        e = 1-x
        i = x*(1-y)
        r = x*y
        return np.array([i, r, e])


class SIR_nonMarkov(SIR_lockdown):
    def __init__(self, growth_rate_init, infectious_period_dist, date_lockdown, pop_size,
                 two_step_measures = False, growth_rate_before_measures = 0, date_of_measures = ''):
#        print('SIR_nonMarkov.__init__')
        assert is_dist(infectious_period_dist) and np.min(infectious_period_dist[:,0] >= 0)
        self.I_dist = infectious_period_dist
        if not hasattr(self, 'N'):
            SIR_lockdown.__init__(self, growth_rate_init, 1, date_lockdown, pop_size,
                                  two_step_measures, growth_rate_before_measures, date_of_measures)
        
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
    
    def run(self, T, dt = .005, record = True):
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
        super().forget()
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
    
class SEIR_nonMarkov(SIR_nonMarkov, SEIR_lockdown):
    def __init__(self, growth_rate_init, EI_period_dist, date_lockdown, pop_size,
                 two_step_measures = False, growth_rate_before_measures = 0, date_of_measures = ''):
#        print('SEIR_nonMarkov.__init__')
        assert is_dist(EI_period_dist, dim = 2) and (np.min(EI_period_dist) >= 0).all()
        self.EI_dist = EI_period_dist
        self.E_dist = self.EI_dist[:,0::2]
        SIR_nonMarkov.__init__(self, growth_rate_init, self.EI_dist[:,1:], date_lockdown, pop_size,
                               two_step_measures, growth_rate_before_measures, date_of_measures)
        # I think I do not need to call SEIR_lockdown_mixed_delays.__init__()
        
    def LaplaceE(self, r):
        return np.sum(self.E_dist[:,1]*np.exp(-r*self.E_dist[:,0]))
    
    def LaplaceEI(self, r):
        return np.sum(self.EI_dist[:,2]*np.exp(-r*(self.EI_dist[:,0]+self.EI_dist[:,1])))
    
    def EE(self):
        return np.sum(self.E_dist[:,1]*self.E_dist[:,0])
        
    def contact_rate(self, r):
        return r/(self.LaplaceE(r)-self.LaplaceEI(r))
    
    def lead_eigen_vect(self, rho = None):
        if rho == None:
            rho = self.r
        l = self.contact_rate(rho)
        i = rho/l
        e = 1-self.LaplaceE(rho)
        r = self.LaplaceEI(rho)
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

class SEIR_varying_inf(SEIR_nonMarkov):
    def __init__(self, growth_rate_init, lambda_bar, date_lockdown, pop_size,
                 two_step_measures = False, growth_rate_before_measures = 0, date_of_measures = ''):
        self.lambda_bar = lambda_bar
        self.I = integrate.quad(self.lambda_bar, 0, np.inf).y
        super().__init__(growth_rate_init, np.array([[1, 1, 1]]), date_lockdown, pop_size,
              two_step_measures, growth_rate_before_measures, date_of_measures)
        
    def contact_rate(self, r):
        return integrate.quad(lambda t: self.lambda_bar(t)*np.exp(-r*t), 0, np.inf).y**-1
    
    def R0(self):
        return self.l*self.I
    
    def contact_rate_R0(self, R0):
        return R0/self.I
    
    def lead_eigen_vect(self, rho = None):
        if rho == None:
            rho = self.r
        pass