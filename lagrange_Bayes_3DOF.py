#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is associated with the paper:
-- A Bayesian framework for discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of 3DOF oscillator" 
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import utils_data
import utils
import sympy as sym
import seaborn as sns
import matplotlib.gridspec as gridspec

np.random.seed(0)

# %%
""" Generating system response """

# The time parameters:
x0 = np.array([0.1, 0, 0.5, 0, 1, 0])
dt, t0, T = 0.001, 0, 1
tparam = [dt, t0, T]
xt, t_eval = utils_data.mdof_system(x0, tparam)

noise = 0.0
xt = xt + noise*np.var(xt[0])*np.random.randn(xt.shape[0],xt.shape[1])

fig1 = plt.figure(figsize=(10,10))
fig1.subplots_adjust(hspace=0.5)
plt.subplot(6,1,1); plt.plot(t_eval, xt[0,:], 'r'); plt.grid(True); plt.margins(0)
plt.subplot(6,1,2); plt.plot(t_eval, xt[1,:]); plt.grid(True); plt.margins(0)
plt.subplot(6,1,3); plt.plot(t_eval, xt[2,:], 'r'); plt.grid(True); plt.margins(0)
plt.subplot(6,1,4); plt.plot(t_eval, xt[3,:]); plt.grid(True); plt.margins(0)
plt.subplot(6,1,5); plt.plot(t_eval, xt[4,:], 'r'); plt.grid(True); plt.margins(0)
plt.subplot(6,1,6); plt.plot(t_eval, xt[5,:]); plt.grid(True); plt.margins(0)

# %%
""" Generating the design matrix """
# Form Lagrangian library:
xvar = np.array([sym.symbols('x'+str(i)) for i in range(1, 6+1)])
D, nd = utils.library_mdof(xvar, polyn=5, harmonic=True)

Rl, dxdt = utils.euler_lagrange_mdof(D, xvar, xt, dt)

# %%
"""
# Gibbs sampling:
"""
# Parameter Initialisation:
MCMC = 1500 # No. of samples in Markov Chain,
burn_in = 500
Z_store, Z_mean, Xi_store, Xi, Xi_sigma = [], [], [], [], []
zstore_all, theta_all, compute_time = [], [], []
for i in range(3):
    print('state-{}'.format(i))
    
    Zs, Zm, th, m, s, zs_all, th_all, c_time = utils.sparse(dxdt[i], Rl[i], nd-1, MCMC, burn_in)
    Z_store.append(Zs)
    Z_mean.append(Zm)
    Xi_store.append(th)
    Xi.append(m)
    Xi_sigma.append(s)
    zstore_all.append(zs_all) 
    theta_all.append(th_all)
    compute_time.append(c_time)

Z_store = np.array(Z_store)
Z_mean = (np.array(Z_mean)).T
Xi_store = np.array(Xi_store)
Xi = (np.array(Xi)).T
Xi_sigma = np.array(Xi_sigma)
zstore_all = np.array(zstore_all)
theta_all = np.array(theta_all)
compute_time = np.mean(np.array(compute_time), axis=0)

xvar_vel = xvar[1::2]
for i in range(3):
    # Z_mean[:,i][np.where(Z_mean[:,i] < 0.8)] = 0    
    Xi[:,i][np.where(Z_mean[:,i] < 0.8)] = 0

Xi_final = np.zeros([nd, 3])
Zmean_final = np.zeros([nd, 3])
for i in range(3):
    Xi_final[:,i] = np.insert(Xi[:,i], np.where(D == xvar_vel[i]**2)[0], 1)
    Zmean_final[:,i] = np.insert(Z_mean[:,i], np.where(D == xvar_vel[i]**2)[0], 1)
Xi_final[11, 2] = 0

Xi_std = np.std(Xi_store,2)/2
Xi_r = Xi/2
    
""" Lagrangian """
xdot = xvar[1::2]
Xi_reduced = utils.nonequal_sum(np.array(Xi_final))
L = np.sum(sym.Array(0.5*np.dot(D,Xi_reduced)))
H = 0
for i in range(len(xdot)):
    H += (sym.diff(L, xdot[i])*xdot[i])
H = H - L
print('Lagrangian: %s, \nHamiltonian: %s' % (L, H))

Lfun = sym.lambdify([xvar], L, 'numpy') 
Hfun = sym.lambdify([xvar], H, 'numpy')

Xi_actual = np.zeros([*Xi_final.shape])
Xi_actual[8,0], Xi_actual[10,1], Xi_actual[12,2] = 1, 1, 1
Xi_actual[7,0], Xi_actual[40,0], Xi_actual[40,1] = -5000, -5000, -5000
Xi_actual[42,1], Xi_actual[42,2] = -5000, -5000
print('Identification Error:', 100*np.linalg.norm(Xi_actual-Xi_final)/np.linalg.norm(Xi_actual))

# %%
""" Hamiltonian """
m1, m2, m3, k1, k2, k3 = 1, 1, 1, 5000, 5000, 5000
H_a = (0.5*m1*xt[1,:]**2 + 0.5*m2*xt[3,:]**2 + 0.5*m3*xt[5,:]**2) + \
    (0.5*k1*xt[0,:]**2 + 0.5*k2*(xt[2,:]-xt[0,:])**2 + 0.5*k3*(xt[4,:]-xt[2,:])**2)
H_i = Hfun(xt)
        
L_a = (0.5*m1*xt[1,:]**2 + 0.5*m2*xt[3,:]**2 + 0.5*m3*xt[5,:]**2) - \
    (0.5*k1*xt[0,:]**2 + 0.5*k2*(xt[2,:]-xt[0,:])**2 + 0.5*k3*(xt[4,:]-xt[2,:])**2)
L_i = Lfun(xt)
    
print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22
t_eval = np.linspace(0,T,len(xt[0]))

fig2 = plt.figure(figsize=(8,5))
plt.plot(t_eval, H_a, 'b', label='Actual')
plt.plot(t_eval, H_i, 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([500,1500])
plt.grid(True)
plt.legend()
plt.margins(0)

