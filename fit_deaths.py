#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:11:33 2020

@author: raphael
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as sp

#import sys
#sys.path.append('/home/raphael/Recherche/Covid19')
import fit_lockdown as lockdown

data = pd.read_csv('donnees-hospitalieres-covid19-2020-05-10-19h00.csv', delimiter = ';')

deaths_early = pd.read_csv('deces_france_0101-1404.csv', index_col = 'jour')

# forget sex
data = data[data['sexe'] == 0]
# remove unused columns
deaths = data.pivot(index = 'jour', columns = 'dep', values = 'dc')

# remove overseas
deaths = deaths.drop(['971', '972', '973', '974', '976'], axis=1)

N_france = 67e6

IDF = ['92', '75', '77', '78', '91', '93', '94', '95']
N_idf = 12.21e6
GrandEst = ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88']
N_GE = 5.55e6
HautsdeFrance = ['02', '59', '60', '62', '80']
N_HdF = 6e6

N_out = N_france - N_idf - N_GE - N_HdF

'''
France = deaths.sum(axis=1)
France = pd.DataFrame(France, columns = ['deces'])
France = pd.concat((deaths_early['2020-02-15':'2020-03-17'], France), axis = 0)
#France['daily'] = np.concatenate(([0], np.diff(France['deces'])))

fit = lockdown.Fitter(France, '2020-03-17', 25)

fit.fit_init('2020-03-01', '2020-03-24')

fit.fit_lockdown('2020-05-10')

fit.plot_fit()
fit.axes.set_title('Cumulative and daily hospital deaths in mainland France')

#fit.compute_SIR(3.5, N_france, 60)
#fit.plot_SIR()
'''

IdF = deaths[IDF].sum(axis=1)
IdF = pd.DataFrame(IdF, columns = ['deces'])
#IdF['daily'] = np.concatenate(([0], np.diff(IdF['deces'])))

fit_idf = lockdown.Fitter(IdF, '2020-03-18', 21)

fit_idf.fit_init('2020-03-19', '2020-03-26')
fit_idf.fit_lockdown('2020-05-10')

fit_idf.plot_fit()
fit_idf.axes.set_title('Cumulative and daily hospital deaths in Ile de France')

lockdown_length = 100
R0_after = 1.2
f = .006
#delays = np.array([[10, .17], [12.5, .18], [15, .19], [17.5, .19], [20, .17], [22.5, .1]])
delays = np.array([[10, .3], [20, .7]])

#alpha = 2.
#x = np.linspace(0,1,50)
#dx = x[1]-x[0]
#beta = np.transpose(np.vstack((x, dx*x**(alpha-1)*(1-x)**(alpha-1)/sp.beta(alpha, alpha))))
#beta[:,1] = beta[:,1]/np.sum(beta[:,1])
#
#modes = np.array([[7, 8, .8], [14, 8, .2]])
#mode1 = np.zeros(np.shape(beta))
#mode1[:,0] = modes[0,0] + modes[0,1]*(beta[:,0]-.5)
#mode1[:,1] = modes[0,2]*beta[:,1]
#mode2 = np.zeros(np.shape(beta))
#mode2[:,0] = modes[1,0] + modes[1,1]*(beta[:,0]-.5)
#mode2[:,1] = modes[1,2]*beta[:,1]
#I_dist = np.vstack((mode1, mode2))

I_dist = np.array([[7, .8], [14, .2]])
g = np.sum(I_dist[:,1]*I_dist[:,0])

EI_dist = np.array([[3, 7, .9], [3, 14, .1]])

#sir = lockdown.SEIR_lockdown_mixed_delays(N_idf, fit_idf.r, fit_idf.rE, f, g, 3, delays)
#sir = lockdown.SIR_nonMarkov(N_idf, fit_idf.r, fit_idf.rE, f, I_dist, delays)
sir = lockdown.SEIR_nonMarkov(N_idf, fit_idf.r, fit_idf.rE, f, EI_dist, delays)
sir.calibrate(fit_idf.deaths_at_lockdown())
#sir.run(300, record = True)
sir.run_full(lockdown_length, 0, R0_after)
sir.plot(S = True)
sir.ax.set_title('Predicted epidemic in Ile de France\ncase fatality rate: %.1f%%' % (100*sir.f))
sir.compute_deaths()
sir.plot_deaths_fit(fit_idf.data)
sir.fig.suptitle('Predicted and observed deaths in Ile de France')
sir.dfit_axs[1].set_yscale('log')
sir.dfit_axs[1].plot(sir.times, sir.daily_deaths[-1]*np.exp(fit_idf.rE*(sir.times-sir.times[-1])))

'''
NordEst = deaths[GrandEst + HautsdeFrance].sum(axis = 1)
fit_NordEst = lockdown.Fitter(NordEst, '2020-03-18', 23)
fit_NordEst.fit_init('2020-03-18', '2020-03-25')
fit_NordEst.fit_lockdown('2020-05-10')
fit_NordEst.plot_fit()
fit_NordEst.axes.set_title('Cumulative and daily hospital deaths in Grand Est and Hauts-de-France')

#lockdown_length = 55
#R0_after = 1.5
#
France_bar_idf = deaths.drop(IDF + GrandEst + HautsdeFrance, axis = 1).sum(axis = 1)
fit = lockdown.Fitter(France_bar_idf, '2020-03-18', 23)
fit.fit_init('2020-03-18', '2020-03-25')
fit.fit_lockdown('2020-05-10')
fit.plot_fit()
fit.axes.set_title('Cumulative and daily hospital deaths in green areas')
#
#sir = lockdown.SIR_lockdown_mixed_delays(54e6, fit.r, fit.rE, f, g, delays)
##sir = lockdown.SIR_nonMarkov(N_france - N_idf, fit.r, fit.rE, f, I_dist, delays)
#sir.calibrate(fit.deaths_at_lockdown())
#sir.run_full(lockdown_length, 200, R0_after)
#sir.plot()
#sir.ax.set_title('Predicted epidemic outside Ile de France')
#print(sir.Z)
#sir.compute_deaths()
#sir.plot_deaths_fit(fit.data)
#sir.fig.suptitle('Predicted and observed deaths outside Ile de France')
'''
'''
paca = deaths[['13', '84', '83', '04', '05', '06']].sum(axis=1)
fit_paca = lockdown.Fitter(paca, '2020-03-18', 26, 3, 3.9, .005)
fit_paca.fit_init('2020-03-18', '2020-03-26')
fit_paca.fit_lockdown('2020-04-27')
fit_paca.plot_fit()

sir_paca = lockdown.SIR_lockdown(5e6, fit_paca.r, fit_paca.rE, .005, 10, 21)
sir_paca.calibrate(fit_paca.deaths_at_lockdown())
sir_paca.run_full(lockdown_length, 200, R0_after)
sir_paca.plot()
sir_paca.ax.set_title('Predicted epidemic in Bouches du Rhone')
print(sir_paca.Z)
'''

#lockdown_length = 55
#R0_before = 3.7
#R0_after = 1.8
#fit_idf.compute_SIR(R0_before, N_idf, lockdown_length)
#fit_idf.ease_lockdown(R0_after, 200)
#fit_idf.plot_SIR()
#fit_idf.sir.ax.vlines(fit_idf.time_before_lockdown + lockdown_length, 0, 1, label = 'May 11th')
#fit_idf.sir.ax.legend(loc='best')
#fit_idf.sir.ax.set_title('Predicted epidemic in Ile de France,\n$R_0$ before lockdown = %.1f, $R_0$ after lockdown = %.1f' % (R0_before, R0_after))
#
#fit_idf.compute_deaths()
#plt.figure(dpi=200)
#plt.plot(fit_idf.sir.times, N_idf*fit_idf.new_deaths)
#plt.plot(fit_idf.sir.times, N_idf*fit_idf.deaths)
#plt.vlines(fit_idf.time_before_lockdown, 0, .1*N_idf)
##plt.plot(fit_idf.data)
#plt.yscale('log')


#t0 = '2020-03-01'
#t_conf = 16
#
#''' params = [r, rE, Dt_conf, mu, sigma] '''
#
#def Fc(t, mu, sigma):
#    return .5*(1-sp.erf((np.log(t)-mu)/(np.sqrt(2)*sigma)))
#
#def deaths(t, params):
#    if t <= t_conf:
#        I = intg.quad(lambda s: params[2]*np.exp(params[0]*(s-t_conf))*Fc(t-s,params[3], params[4]), -np.inf, t)[0]
#    else:
#        I1 = intg.quad(lambda s: params[2]*np.exp(params[0]*(s-t_conf))*Fc(t-s,params[3], params[4]), -np.inf, t_conf)[0]
#        I2 = intg.quad(lambda s: params[2]*np.exp(params[1]*(s-t_conf))*Fc(t-s,params[3], params[4]), t_conf, t)[0]
#        I = I1 + I2
#    return I
#
#def death_curve(params):
#    times = np.arange(np.size(true_values))
#    death = np.zeros(np.size(times))
#    for t in times:
#        death[t] = deaths(t, params)
#    return death
#
#scale = [.01, .01, 100, 1, 1]
##scale = [.1, .1, 100]
#init_params = np.array([0.23, 0.045, 100, 2, np.sqrt(2)])/scale
##init_params = np.array([0.23, 0.045, 300])/scale
##musigma = [2, np.sqrt(2)]
#true_values = France[t0:].values
#true_values = np.reshape(true_values, (np.size(true_values)))
#
#def fit(params):
##    params = np.concatenate((scale*np.abs(params), musigma))
#    params = np.abs(params)
#    death = death_curve(params)
#    fit = np.sum((np.log(death)-np.log(true_values))**2)
##    print(fit, params)
#    return fit
#
#result2 = optim.minimize(fit, init_params)
#
##estimate = np.concatenate((scale*np.abs(result2.x), musigma))
#estimate = scale*np.abs(result2.x)
#print(estimate)
#    
#time = np.arange(np.size(true_values))
#plt.plot(time, true_values)
#plt.plot(time, death_curve(estimate))
#plt.yscale('log')
