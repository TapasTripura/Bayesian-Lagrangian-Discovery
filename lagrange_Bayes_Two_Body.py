#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is associated with the paper:
-- A Bayesian framework for discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code is for "discovering Lagrangian of Two-body problem" 

The two-body problem is to predict the motion of two massive objects which are
 abstractly viewed as point particles. 
 The problem assumes that the two objects interact only with one another; the
 only force affecting each object arises from the other one, and all other
 objects are ignored.
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

# The parameters:
G = 9.81
m1 = 1
m2 = 1

x0 = [1, 0, 0, 1, 0, 0, 0, 0]
dt, t0, T = 0.001, 0, 1
tparam = [dt, t0, T]
xt, t_eval = utils_data.twobody(x0, tparam, G, m1, m2)

# %%
r1d = (xt[0,:]**2 + xt[1,:]**2)**(1/2)
r2d = (xt[4,:]**2 + xt[5,:]**2)**(1/2)
r1dot = (xt[2,:]**2 + xt[3,:]**2)**(1/2)
r2dot = (xt[6,:]**2 + xt[7,:]**2)**(1/2)
V = (-G * m1 * m2) / ( ((xt[0,:] - xt[4,:])**2 + (xt[1,:] - xt[5,:])**2)**(1/2) )
L = 0.5 * m1 * r1dot**2 + 0.5 * m2 * r2dot**2 + V

xt_derived = np.stack((xt[0,:], r1dot, xt[4,:], r2dot))

# %%
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
plt.subplots_adjust(hspace=0.3)

ax[0].plot(xt[0,:],xt[1,:],'k')
ax[0].plot(xt[4,:],xt[5,:],'r')
ax[0].grid(True, alpha=0.35)

ax[1].plot(t_eval,L)
ax[1].set_ylim([-20,0])
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Hamiltonian ($\mathcal{H}$)')
ax[1].grid(True, alpha=0.35)

# %%
""" Generating the design matrix """
# Form Lagrangian library:
xvar = np.array([sym.symbols('x'+str(i)) for i in range(1, 4+1)])
D, _ = utils.library_mdof(xvar, polyn=3)

Rl, dxdt = utils.euler_lagrange_mdof(D, xvar, xt_derived, dt)

# Cyclic components, for two/three-body problems:
r1, r2 = sym.symbols(['r1', 'r2'])

V_grav = np.divide(1, r1d, where=r1d!=0)[:,None]
Rl[0] = np.append(Rl[0], V_grav, axis=1)
Rl[1] = np.append(Rl[1], V_grav, axis=1)
D = np.append(D, 1/r1)

V_grav = np.divide(1, r1d**2, where=r1d!=0)[:,None]
Rl[0] = np.append(Rl[0], V_grav, axis=1)
Rl[1] = np.append(Rl[1], V_grav, axis=1)
D = np.append(D, 1/r1**2)

V_grav = np.divide(1, r2d, where=r2d!=0)[:,None]
Rl[0] = np.append(Rl[0], V_grav, axis=1)
Rl[1] = np.append(Rl[1], V_grav, axis=1)
D = np.append(D, 1/r2)

V_grav = np.divide(1, r2d**2, where=r2d!=0)[:,None]
Rl[0] = np.append(Rl[0], V_grav, axis=1)
Rl[1] = np.append(Rl[1], V_grav, axis=1)
D = np.append(D, 1/r2**2)

V_grav = (1 / (((xt[0,:] - xt[4,:])**2 + (xt[1,:] - xt[5,:])**2)**(2/2)))[:,None]
Rl[0] = np.append(Rl[0], V_grav, axis=1)
Rl[1] = np.append(Rl[1], V_grav, axis=1)
D = np.append(D, 1/(r1-r2))

V_grav = ((xt[1,:] - xt[5,:]) / (((xt[0,:] - xt[4,:])**2 + (xt[1,:] - xt[5,:])**2)**(3/2)))[:,None]
Rl[0] = np.append(Rl[0], V_grav, axis=1)
Rl[1] = np.append(Rl[1], V_grav, axis=1)
D = np.append(D, 1/(r1-r2)**2)

V_grav = (9*(xt[1,:] - xt[5,:])**2 / (((xt[0,:] - xt[4,:])**2 + (xt[1,:] - xt[5,:])**2)**(4/2)))[:,None]
Rl[0] = np.append(Rl[0], V_grav, axis=1)
Rl[1] = np.append(Rl[1], V_grav, axis=1)
D = np.append(D, 1/(r1-r2)**3)

V_grav = (16*(xt[1,:] - xt[5,:])**3 / (((xt[0,:] - xt[4,:])**2 + (xt[1,:] - xt[5,:])**2)**(5/2)))[:,None]
Rl[0] = np.append(Rl[0], V_grav, axis=1)
Rl[1] = np.append(Rl[1], V_grav, axis=1)
D = np.append(D, 1/(r1-r2)**4)

V_grav = (25*(xt[1,:] - xt[5,:])**4 / (((xt[0,:] - xt[4,:])**2 + (xt[1,:] - xt[5,:])**2)**(6/2)))[:,None]
Rl[0] = np.append(Rl[0], V_grav, axis=1)
Rl[1] = np.append(Rl[1], V_grav, axis=1)
D = np.append(D, 1/(r1-r2)**5)

V_grav = (36*(xt[1,:] - xt[5,:])**5 / (((xt[0,:] - xt[4,:])**2 + (xt[1,:] - xt[5,:])**2)**(7/2)))[:,None]
Rl[0] = np.append(Rl[0], V_grav, axis=1)
Rl[1] = np.append(Rl[1], V_grav, axis=1)
D = np.append(D, 1/(r1-r2)**6)

nd = len(D)

# %%
"""
# Gibbs sampling:
"""
# Parameter Initialisation:
MCMC = 2000 # No. of samples in Markov Chain,
burn_in = 1000
Z_store, Z_mean, Xi_store, Xi, Xi_sigma = [], [], [], [], []
zstore_all, theta_all, compute_time = [], [], []
for i in range(2):
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
for i in range(2):
    # Z_mean[:,i][np.where(Z_mean[:,i] < 0.8)] = 0    
    Xi[:,i][np.where(Z_mean[:,i] < 0.8)] = 0

Xi_final = np.zeros([nd, 2])
Zmean_final = np.zeros([nd, 2])
for i in range(2):
    Xi_final[:,i] = np.insert(Xi[:,i], np.where(D == xvar_vel[i]**2)[0], 1)
    Zmean_final[:,i] = np.insert(Z_mean[:,i], np.where(D == xvar_vel[i]**2)[0], 1)

Xi_std = (np.std(Xi_store,2)/2).T
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
Xi_actual[6,0], Xi_actual[8,1] = 1, 1
Xi_actual[21,0], Xi_actual[21,1] = -9.81, -9.81
print('Identification Error:', 100*np.linalg.norm(Xi_actual-Xi_final)/np.linalg.norm(Xi_actual))

# %%
""" Hamiltonian """
V_a = (-G * m1 * m2) / ( ((xt[0,:] - xt[4,:])**2 + (xt[1,:] - xt[5,:])**2)**(1/2) )
L_a = 0.5 * m1 * r1dot**2 + 0.5 * m2 * r2dot**2 - V_a
H_a = 0.5 * m1 * r1dot**2 + 0.5 * m2 * r2dot**2 + V_a

V_i = np.mean(Xi_final[21]) / ( ((xt[0,:] - xt[4,:])**2 + (xt[1,:] - xt[5,:])**2)**(1/2) )
L_i = 0.5 * m1 * r1dot**2 + 0.5 * m2 * r2dot**2 - V_i
H_i = + 0.5 * m1 * r1dot**2 + 0.5 * m2 * r2dot**2 + V_i
  
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
plt.ylim([-20,20])
plt.grid(True)
plt.legend()
plt.margins(0)

