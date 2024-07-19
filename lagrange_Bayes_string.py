#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is associated with the paper:
-- A Bayesian framework for discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of String" 
"""

import numpy as np
import matplotlib.pyplot as plt
import utils_data
import utils
import sympy as sym

np.random.seed(0)

# %%
""" Generating system response """

# The time parameters:
dx = 0.1        # Spacing of points on string
dt = 0.001      # Size of time step
c = 10          # Speed of wave propagation
print('r-{}'.format(c*(dt/dx)))
L = 1           # Length of string
stopTime = 1    # Time to run the simulation
t  = np.arange(0, stopTime+dt, dt)
mesh = np.arange(0, L+dt, dx)

xt = utils_data.string(L,stopTime,c,dx,dt)

# %%
Xi_actual = np.zeros((54,10))
for i in range(1,Xi_actual.shape[1]-1):
    Xi_actual[i+1, i] = 1
    Xi_actual[i+21, i] = -100
    Xi_actual[i+22, i] = -100

# %%
""" Add Noise, if required """
noise = 0.0
xt[2:-2,:] = xt[2:-2,:] + noise*np.min(np.std(xt,1)[2:-2])*np.random.randn(xt[2:-2].shape[0],xt.shape[1])

dis = xt[::2]
vel = xt[1::2]

# %%
fig1 = plt.figure(figsize=(18,10))
plt.subplot(2,1,1); plt.imshow(dis, cmap='nipy_spectral', aspect='auto')
plt.subplot(2,1,2); plt.imshow(vel, cmap='nipy_spectral', aspect='auto')

print("Response generated...")

# %%
l = 1000    # To plot the first 1000 time points, 
fig11 = plt.figure(figsize=(10,8))
ax = plt.axes(projection ='3d')
a, b = np.meshgrid(t[:-1], mesh)
surf = ax.plot_surface(a[:,:l], b[:,:l], dis[:,:l], cmap='ocean', antialiased=False)
ax.view_init(30,-130)
fig11.colorbar(surf, shrink=0.85, aspect=20)

ax.set_xlabel('Time (s)', labelpad = 20, fontweight='bold');
ax.set_ylabel('x', labelpad = 20, fontweight='bold');
ax.set_zlabel('u(x,t)', labelpad = 10, fontweight='bold')
plt.margins(0)

# %%
""" Generating the design matrix """
# Form Lagrangian library:
xvar = [sym.symbols('x'+str(i)) for i in range(1, 2*int(L/dx)+1)]
D, nd = utils.library_pde(xvar, Type='order1', dx=dx, polyn=4)

Rl, dxdt = utils.euler_lagrange_mdof(D, xvar, xt, dt)

# %%
"""
# Gibbs sampling:
"""
# Parameter Initialisation:
data = dxdt
MCMC = 2000 # No. of samples in Markov Chain,
burn_in = 1000
Z_store, Z_mean, Xi_store, Xi, Xi_sigma = [], [], [], [], []
for kk in range(int(len(xvar)/2)):
    print('Element- ', kk)
    if len(np.where( data[kk] != 0)[0]) < 5: # to check, if the vector is full of zeros
        Zs, Zm, th, m, s = np.zeros([nd-1,MCMC-burn_in]), np.zeros(nd-1), \
            np.zeros([nd-1,MCMC-burn_in]), np.zeros(nd-1), np.zeros([nd-1,nd-1])
    else:
        Zs, Zm, th, m, s, _, _, _ = utils.sparse(dxdt[kk], Rl[kk], nd-1, MCMC, burn_in)
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

xvar_vel = xvar[1::2]
for i in range(len(xvar_vel)):
    Z_mean[:,i][np.where(Z_mean[:,i] < 0.9)] = 0    
    Xi[:,i][np.where(Z_mean[:,i] < 0.9)] = 0

Xi_final = np.zeros([nd, int(len(xvar)/2)])
Zmean_final = np.zeros([nd, int(len(xvar)/2)])
for i in range(len(xvar_vel)):
    if len(np.where( data[i] != 0)[0]) < 5:
        Xi_final[:,i] = np.insert(Xi[:,i], np.where(D == xvar_vel[i]**2)[0], 0)
        Zmean_final[:,i] = np.insert(Z_mean[:,i], np.where(D == xvar_vel[i]**2)[0], 0)
    else:
        Xi_final[:,i] = np.insert(Xi[:,i], np.where(D == xvar_vel[i]**2)[0], 1)
        Zmean_final[:,i] = np.insert(Z_mean[:,i], np.where(D == xvar_vel[i]**2)[0], 1)
    
predict = np.sqrt( np.abs( Xi_final[22,1] ) )
rel_error = 100*np.abs(predict-c)/c
print("Actual: %d, Predicted: %0.4f, Relative error: %0.4f percent." % (c,predict,rel_error))

Xi_std = np.std(Xi_store,2).T/2
Xi_r = Xi/2

""" Lagrangian """
L = np.dot(D,Xi_final)

# %%
""" Hamiltonian """
const = c**2/dx**2/2
T = np.sum(vel[:-2]**2, axis=0)
V = const*(np.sum(np.diff(dis,axis=0)**2, axis=0))

H_a = 0.5*T + V
L_a = 0.5*T - V
    
const_i = predict**2/dx**2/2
T_i = np.sum(vel[:-2]**2, axis=0)
V_i = const_i*(np.sum(np.diff(dis,axis=0)**2, axis=0))

H_i = 0.5*T_i + V_i
L_i = 0.5*T_i - V_i

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
plt.ylim([0,200])
plt.grid(True)
plt.legend()
plt.margins(0)

