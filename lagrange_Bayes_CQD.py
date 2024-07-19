#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is associated with the paper:
-- A Bayesian framework for discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of cubic-quintic-Duffing oscillator" 
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import utils_data
import utils
import sympy as sym
import seaborn as sns
import pandas as pd

np.random.seed(3)

# %%
""" Generating system response """

# The time parameters:
x0 = np.array([0.55, 0])
dt, t0, T = 0.0005, 0, 0.5
tparam = [dt, t0, T]
a, b, c = 1000,5000,90000
params = [a, b, c]
xt, t_eval = utils_data.duffing_quintic(x0, tparam, params)

noise = 0.01
xt = xt + noise*np.var(xt[0])*np.random.randn(xt.shape[0],xt.shape[1])

fig1 = plt.figure(figsize=(10,8))
plt.subplot(2,1,1); plt.plot(t_eval, xt[0,:]); plt.grid(True); plt.margins(0)
plt.subplot(2,1,2); plt.plot(t_eval, xt[1,:]); plt.grid(True); plt.margins(0)

# %%
""" Generating the design matrix """

xvar = [sym.symbols('x'), sym.symbols('y')]
D, nd = utils.library_sdof(xvar,polyn=10,harmonic=True,harmonic_interaction=True)

Rl, dxdt = utils.euler_lagrange_mdof(D,xvar,xt,dt)

# %%
"""
# Gibbs sampling:
"""
# Parameter Initialisation:
MCMC = 2000 # No. of samples in Markov Chain,
burn_in = 1000

# First iteration:
Z_store, Z_mean, Xi_store, Xi, Xi_sigma, zstore_all, theta_all, _ = utils.sparse(dxdt[0], Rl[0], nd-1, MCMC, burn_in)

PIP = 0.8
inds = np.where(Z_mean > PIP)
Rl_inds = Rl[0][:, inds[0]]
# Second iteration:
Z_sf, Z_mf, Xi_sf, Xi_f, Xi_sigf, zstore_af, theta_af, _ = utils.sparse(dxdt[0], Rl_inds, len(inds[0]), MCMC, burn_in)

Z_store[inds[0], :] = Z_sf
Z_mean[inds] = Z_mf
Xi_store[inds[0], :] = Xi_sf
Xi[inds] = Xi_f
Xi_sigma[inds][:, inds[0]] = Xi_sigf
zstore_all[inds[0], :] = zstore_af
theta_all[inds[0], :] = theta_af

Xi[np.where(Z_mean < PIP)] = 0

Xi = np.insert(Xi,np.where(D == xvar[1]**2)[0], 1, axis=0)
Z_mean = np.insert(Z_mean,np.where(D == xvar[1]**2)[0], 1, axis=0)
Xi_std = np.std(Xi_store,1)/2

print(Xi)

""" Lagrangian """
L = 0.5*np.dot(D,Xi)
Xi_org = np.zeros(len(D))
Xi_org[1] = 1
Xi_org[3] = -1000
Xi_org[5] = -2500
Xi_org[7] = -30000
print('Identification Error:', 100*np.mean((Xi-Xi_org)**2)/np.mean(Xi_org**2))

# %%
""" Hamiltonian """
H_a = (1/2)*xt[1,:]**2+(1/2)*a*xt[0,:]**2+(1/4)*b*xt[0,:]**4+(1/6)*c*xt[0,:]**6
H_i = (1/2)*xt[1,:]**2+(1/2)*np.abs(Xi[3])*xt[0,:]**2+(1/2)*np.abs(Xi[5])*xt[0,:]**4+(1/2)*np.abs(Xi[7])*xt[0,:]**6

L_a = (1/2)*xt[1,:]**2-((1/2)*a*xt[0,:]**2+(1/4)*b*xt[0,:]**4+(1/6)*c*xt[0,:]**6)
L_i = (1/2)*xt[1,:]**2-((1/2)*np.abs(Xi[3])*xt[0,:]**2+(1/2)*np.abs(Xi[5])*xt[0,:]**4+(1/2)*np.abs(Xi[7])*xt[0,:]**6)

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

fig2 = plt.figure(figsize=(8,5))
plt.plot(t_eval, H_a, 'b', label='Actual')
plt.plot(t_eval, H_i, 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([1e-3,1e6])
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.margins(0)
