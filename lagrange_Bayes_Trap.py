#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is associated with the paper:
-- A Bayesian framework for discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of Penning Trap" 
"""

import numpy as np
import matplotlib.pyplot as plt
import utils_data
import utils
import sympy as sym
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18

np.random.seed(0)

# %%
""" Generating system response """

# The time parameters:
x0 = np.array([1e-2, 0, 1e-2, 0, 1e-1, 0])
dt, t0, T = 0.001, 0, 1
tparam = [dt, t0, T]
params = [100,10]
xt, t_eval = utils_data.penning_trap(x0, params, tparam)

noise = 0.0
xt = xt + noise*np.std(xt[0])*np.random.randn(xt.shape[0],xt.shape[1])

# %%
fig1 = plt.figure(figsize=(14,6))
fig1.subplots_adjust(hspace=0.5)

ax = fig1.add_subplot(1, 2, 1, projection='3d')
ax.scatter3D(xt[0,:], xt[2,:], xt[4,:], c=xt[4,:], cmap='viridis');
ax.set_title('(a) Harmonic oscillator', fontweight='bold')
ax.set_xlabel('$x(t)$', labelpad = 10, fontweight='bold', color='m'); 
ax.set_zlabel('Time (s)', labelpad = 10, fontweight='bold', color='m'); 
ax.set_ylabel('$\dot{x}(t)$', labelpad = 10, fontweight='bold', color='m')
plt.legend(['Truth', 'Discovered'], ncol=2, loc=1, bbox_to_anchor=(1,0.95), columnspacing=0.5,
           handletextpad=0.1, handlelength=0.8, borderpad=0.2, frameon=1,
           fancybox=1, shadow=0, edgecolor=None)
plt.margins(0)

ax = fig1.add_subplot(1, 2, 2, projection='3d')
ax.plot3D(xt[0,:], xt[2,:], xt[4,:], 'r', linewidth=2);
ax.set_title('(b) Forced oscillator', fontweight='bold')
ax.set_xlabel('$x(t)$', labelpad = 10, fontweight='bold', color='m'); 
ax.set_zlabel('Time (s)', labelpad = 10, fontweight='bold', color='m'); 
ax.set_ylabel('$\dot{x}(t)$', labelpad = 10, fontweight='bold', color='m')
plt.legend(['Truth', 'Discovered'], ncol=2, loc=1, bbox_to_anchor=(1,0.95), columnspacing=0.5,
           handletextpad=0.1, handlelength=0.8, borderpad=0.2, frameon=1,
           fancybox=1, shadow=0, edgecolor=None)
plt.margins(0)

# %%
""" Generating the design matrix """
# Form Lagrangian library:
xvar = [sym.symbols('x'+str(i)) for i in range(1, 6+1)]
D, nd = utils.library_sdof(xvar, polyn=2, harmonic=True)

Rl, dxdt = utils.euler_lagrange_mdof(D, xvar, xt, dt)

# %%
"""
# Gibbs sampling:
"""
# Parameter Initialisation:
MCMC = 2000 # No. of samples in Markov Chain,
burn_in = 1000
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
    Z_mean[:,i][np.where(Z_mean[:,i] < 0.75)] = 0    
    Xi[:,i][np.where(Z_mean[:,i] < 0.75)] = 0

Xi_final = np.zeros([nd, 3])
Zmean_final = np.zeros([nd, 3])
for i in range(3):
    Xi_final[:,i] = np.insert(Xi[:,i], np.where(D == xvar_vel[i]**2)[0], 1)
    Zmean_final[:,i] = np.insert(Z_mean[:,i], np.where(D == xvar_vel[i]**2)[0], 1)
    
data = { 'Lagrangian ':D.astype('U'),
         'Theta 0':Xi_final[:,0], 
         'Theta ':Xi_final[:,1], 
         'Theta 2':Xi_final[:,2], 
         }
df = pd.DataFrame(data)
print(df)

if noise == 0:
    Xi_actual = np.zeros((25, 3))
    Xi_actual[1,0], Xi_actual[2,1], Xi_actual[3,2] = 1, 1, 1
    Xi_actual[7,0], Xi_actual[8,1], Xi_actual[9,2] = 50, 50, -100
    Xi_actual[15,0], Xi_actual[12,1] = 200, -200
else:
    Xi_actual = np.zeros((25, 3))
    Xi_actual[1,0], Xi_actual[2,1], Xi_actual[3,2] = 1, 1, 1
    Xi_actual[7,0], Xi_actual[8,1], Xi_actual[9,2] = 50, 50, -100
    Xi_actual[15,1], Xi_actual[12,0] = -200, 200

error = 100*(np.linalg.norm(Xi_actual-Xi_final)/np.linalg.norm(Xi_actual))
print('error={}'.format(error))

# %%
""" Lagrangian """
xdot = xvar[1::2]
L = 0.5*np.dot(D,Xi_final).sum()
H = 0
for i in range(len(xdot)):
    H += (sym.diff(L, xdot[i])*xdot[i])
H = H - L
print('Lagrangian: %s, \n\nHamiltonian: %s' % (L, H))
Lfun = sym.lambdify([xvar], L, 'numpy') 
Hfun = sym.lambdify([xvar], H, 'numpy')

# %%
""" Hamiltonian """
H_a = 0.5*(xt[1,:]**2 + xt[3,:]**2 + xt[5,:]**2)  \
        - 0.25*params[1]**2*(xt[0,:]**2 + xt[2,:]**2 - 2*xt[4,:]**2)   \
            -0.5*params[0]*(xt[0,:]*xt[3,:] - xt[1,:]*xt[2,:])  \
                +0.5*params[0]*(xt[0,:]-xt[2,:])
H_i = Hfun(xt)

L_a = 0.5*xt[1,:]**2 + 0.5*xt[3,:]**2 + 0.5*xt[5,:]**2  \
        + 0.25*params[1]**2*(xt[0,:]**2 + xt[2,:]**2 - 2*xt[4,:]**2)  \
            +params[0]*(+xt[0,:]*xt[3,:] - xt[1,:]*xt[2,:])
L_i = Lfun(xt)

print('Percentage Hamiltonian error: %0.4f' % (100*np.mean(np.abs(H_a-H_i)/np.abs(H_a))) )
print('Percentage Lagrange error: %0.4f' % (100*np.mean(np.abs(L_a-L_i)/np.abs(L_a))) )

""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22

fig2 = plt.figure(figsize=(8,5))
plt.plot(t_eval, H_a, 'b', label='Actual')
plt.plot(t_eval, H_i, 'r--', linewidth=2, label='Identified')
plt.xlabel('Time (s)')
plt.ylabel('Hamiltonian (Energy)')
plt.ylim([0,1e-2])
plt.grid(True)
plt.legend()
plt.margins(0)
