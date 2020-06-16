#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:39:47 2020

@author: raphael
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

def is_dist(dist, dim = 1):
    assert type(dim) == int
    return np.size(dist, axis = 1) == dim + 1 and np.abs(np.sum(dist[:,-1])-1) < 1e-8 and np.min(dist) >= 0

def product_dist(dist1, dist2):
    assert is_dist(dist1) and is_dist(dist2)
    n = np.size(dist1, axis = 0)
    m = np.size(dist2, axis = 0)
    pdist = np.zeros((n*m, 3))
    pdist[:,0] = np.tile(dist1[:,0], m)
    pdist[:,1] = np.repeat(dist2[:,0], n)
    pdist[:,2] = np.tile(dist1[:,1], m)*np.repeat(dist2[:,1], n)
    assert is_dist(pdist, dim = 2)
    return pdist

def beta_dist(alpha, beta, n = 10):
    assert alpha > 0 and beta > 0
    assert type(n)==int and n > 2
    dist = np.zeros((n,2))
    dist[:,0] = np.linspace(0, 1, n, endpoint=False)
    cumdist = np.concatenate((sp.betainc(alpha, beta, dist[:,0]), [1]))
    dist[:,1] = np.diff(cumdist)
    return dist

def lognormal_dist(mu, sigma, n = 10):
    assert sigma > 0
    mu = np.abs(mu)
    dist = np.zeros((n, 2))
    X = mu + sigma*2*np.sqrt(2)
    dist[:,0] = np.concatenate(([0], np.exp(np.linspace(-5, X, n-1))))
    cumdist = np.concatenate((.5*(1+sp.erf((np.log(dist[:,0])-mu)/(sigma*np.sqrt(2)))), [1]))
    dist[:,1] = np.diff(cumdist)
    return dist

def gamma_dist(k, theta, n = 100):
    assert k > 0 and theta > 0
    assert type(n)==int and n > 2
    dist = np.zeros((n,2))
    dist[:,0] = np.linspace(0, 10*k*theta, n, endpoint = False)
    cumdist = np.concatenate((sp.gammainc(k, dist[:,0]/theta), [1]))
    dist[:,1] = np.diff(cumdist)
    return dist

def convol(dist1, dist2):
    assert is_dist(dist1) and is_dist(dist2)
    pdist = product_dist(dist1, dist2)
    pdist[:,0] += pdist[:,1]
    return unique(pdist[:,0::2])

def unique(dist):
    assert is_dist(dist)
    values, indices = np.unique(dist[:,0], return_inverse = True)
    udist = np.zeros((np.size(values), 2))
    udist[:,0] = values
    for i in np.arange(np.size(indices)):
        udist[indices[i],1] += dist[i,1]
    return udist

def EI_dist_covid(p_reported, fixed_E = True, n = 10):
    assert p_reported >= 0 and p_reported <= 1
    if fixed_E:
        E_dist = np.array([[3, 1]])
    else:
        E_dist = [2, 0] + [2, 1]*beta_dist(2, 2, n)
    I_dist = np.concatenate(([2, p_reported]*([3, 0] + beta_dist(2, 2, n)),
             [1, 1-p_reported]*([8, 0] + [4, 1]*beta_dist(2, 2, n))), axis = 0)
    return product_dist(E_dist, I_dist)

def delay_hosp_covid(n = 20):
    return [7, 0] + [10, 1]*beta_dist(1.5, 1.2, n)

def delay_death_covid(n = 20):
    delay_hosp_to_death = [.5, 0] + [10.5, 1]*beta_dist(1.5, 1.5, n)
    return convol(delay_hosp_covid(n), delay_hosp_to_death)

def delay_hosp_death_covid(mean_hosp, offset_hosp, mean_death, offset_death, n = 20):
    assert mean_hosp > offset_hosp
    assert mean_death > offset_death
    delay_hosp = [offset_hosp, 0] + [(mean_hosp-offset_hosp)*2, 1]*beta_dist(2, 2, n)
    delay_hosp_to_death = [offset_death, 0] + [(mean_death-offset_death)*2, 1]*beta_dist(2, 2, n)
    delay_death = convol(delay_hosp, delay_hosp_to_death)
    return delay_hosp, delay_death

def R0(rho, EI_dist):
    return rho*np.sum(EI_dist[:,1]*EI_dist[:,2])/(
        np.sum(EI_dist[:,2]*np.exp(-rho*EI_dist[:,0])) - np.sum(EI_dist[:,2]*np.exp(-rho*(EI_dist[:,0]+EI_dist[:,1]))))

class Dist(object):
    def __init__(self, values, probas):
        if np.size(np.shape(values)) == 1:
            self.dim = 1
        else:
            assert np.size(np.shape(values)) == 2
            self.dim = np.size(values, axis = 1)
        self.n = np.size(values, axis = 0)
        self.values = values
        self.probas = probas
    
    def _set_p(self, probas):
        assert np.size(probas) == self.n
        assert np.min(probas) >= 0 and np.abs(np.sum(probas)-1) < 1e-8
        self._probas = probas
    def _get_p(self):
        return self._probas
    probas = property(_get_p, _set_p)
    
    def _set_v(self, values):
        assert np.shape(values) == (self.n, self.m)
        self._values = values
    def _get_v(self):
        return self._values
    values = property(_get_v, _set_v)
    
    def E(self):
        if self.dim == 1:
            return np.sum(self.values*self.probas)
        else:
            E = np.zeros(self.dim)
            for i in np.arange(self.dim):
                E[i] = np.sum(self.values[:,i]*self.probas)
            return E
    
    def Laplace(self, r):
        if self.dim == 1:
            return np.sum(self.probas*np.exp(r*self.values))
        else:
            L = np.zeros(self.dim)
            for i in np.arange(self.dim):
                L[i] = np.sum(self.probas*np.exp(r*self.values[:,i]))
            return L
    
    def __add__(self, x):
        self.values += x
        return self
    
    def __matmul__(self, dist2):
        assert isinstance(dist2, Dist)
        pass