#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is associated with the paper:
-- A Bayesian framework for discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of Euler-Bernoulli Beam" 
"""

import numpy as np
import matplotlib.pyplot as plt
import utils_data
import utils
import sympy as sym

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

np.random.seed(0)

# %%
""" Generating system response """

# The time parameters:
Ne = 10
dt = 0.0001  # Size of time step
L = 1 # Length of string
T = 0.1 # Time to run the simulation
rho = 7850
b, d = 0.02, 0.001
A = b*d
E = 2e11
I = (b*d**3)/12 
params = [rho, b, d, A, L, E, I]
c = (E*I)/(rho*A)
print('Wave coefficient-{}'.format(c))

t  = np.arange(0, T+dt, dt)
mesh = np.linspace(0, L, Ne)

dis, vel, acc = utils_data.cantilever(params,T,dt,Ne)

xt = np.zeros([2*dis.shape[0],dis.shape[1]])
xt[::2] = dis
xt[1::2] = vel

# %%
Xi_actual = np.zeros((23,10))
for i in range(Xi_actual.shape[1]):
    Xi_actual[i+1, i] = 1
    Xi_actual[i+11, i] = -c
    Xi_actual[i+12, i] = -c
    Xi_actual[i+13, i] = -c

# %%
fig1 = plt.figure(figsize=(10,6))
plt.subplot(3,1,1); plt.imshow(dis, cmap='nipy_spectral', aspect='auto')
plt.subplot(3,1,2); plt.imshow(vel, cmap='nipy_spectral', aspect='auto')
plt.subplot(3,1,3); plt.imshow(acc, cmap='nipy_spectral', aspect='auto')

# %%
""" Generating the design matrix """
# Form Lagrangian library:
xvar = [sym.symbols('x'+str(i)) for i in range(1, 2*int(Ne)+1)]
D, nd = utils.library_pde(xvar, Type='order2', dx=L/Ne, polyn=2)

Rl, dxdt = utils.euler_lagrange_mdof(D, xvar, xt, dt)
dxdt = dxdt*0.5

# %%
"""
# Gibbs sampling:
"""
# Parameter Initialisation:
MCMC = 10000 # No. of samples in Markov Chain,
burn_in = 5000
Z_store, Z_mean, Xi_store, Xi, Xi_sigma = [], [], [], [], []
for kk in range(int(len(xvar)/2)):
    print('Element- ', kk)
    
    Zs, Zm, th, m, s = utils.sparse(dxdt[kk], Rl[kk], nd-1, MCMC, burn_in)
    Z_store.append(Zs)
    Z_mean.append(Zm)
    Xi_store.append(th)
    Xi.append(m)
    Xi_sigma.append(s)

Z_store = np.array(Z_store)
Z_mean = (np.array(Z_mean)).T
Xi_store = np.array(Xi_store)
Xi = (np.array(Xi)).T
Xi_sigma = np.array(Xi_sigma)

for i in range(int(len(xvar)/2)):
    Z_mean[:,i][np.where(Z_mean[:,i] < 0.7)] = 0    
    Xi[:,i][np.where(Z_mean[:,i] < 0.7)] = 0

xvar_vel = xvar[1::2]
Xi_final = np.zeros([nd, int(len(xvar)/2)])
for kk in range(len(xvar_vel)):
    if len(np.where( dxdt[kk] != 0)[0]) < 5:
        Xi_final[:,kk] = np.insert(Xi[:,kk], np.where(D == xvar_vel[kk]**2)[0], 0)
    else:
        Xi_final[:,kk] = np.insert(Xi[:,kk], np.where(D == xvar_vel[kk]**2)[0], 1)

predict = np.abs(Xi_final[16,5])
rel_error = 100*np.abs(predict-c)/c
print("Actual: %d, Predicted: %0.4f, Relative error: %0.4f percent." % (c,predict,rel_error))

# %%
""" Hamiltonian """
T = 0.5*np.sum(vel[:-2,:]**2, axis=0)
V = 0.5*(c/(1/Ne)**4)*np.sum((-np.diff(dis[:-1,:], axis=0) + np.diff(dis[1:,:], axis=0))**2, axis=0)
    
H_a = 0.5*T + 0.5*V
L_a = 0.5*T - 0.5*V
    
T_i = 0.5*np.sum(vel[:-1,:]**2, axis=0)
V_i = 0.5*(predict/(1/Ne)**4)*np.sum((-np.diff(dis[:-1,:], axis=0) + np.diff(dis[1:,:], axis=0))**2, axis=0)

H_i = 0.5*T_i + 0.5*V_i
L_i = 0.5*T_i - 0.5*V_i

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/H_a)) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

fig2 = plt.figure(figsize=(8,6))
plt.plot(t, H_a, 'b', label='Actual')
plt.plot(t, H_i, 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([5e4,7e4])
plt.grid(True)
plt.legend()
plt.margins(0)

