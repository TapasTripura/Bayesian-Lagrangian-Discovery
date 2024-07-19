#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:31:09 2024

@author: user
"""

import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

def hamiltonian_fn(coords):
    a, b, c = 1000, 5000, 90000
    q, p = np.split(coords,2)
    H = 0.5*p**2 + 0.5*a*q**2 + 0.25*b*q**4 + (1/6)*c*q**6 # CQD hamiltonian (nonlinear oscillator)
    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dpdt, dqdt = np.split(dcoords,2)
    S = np.concatenate([dqdt, -dpdt], axis=-1)
    return S

t_span=[0,0.5]
timescale=1000
radius=None
y0=None
noise_std=0.0
t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
# get initial state
if y0 is None:
    y0 = np.random.uniform(0.1, 0.5, size=2)
if radius is None:
    radius = np.random.rand()*0.5 + 0.1 # sample a range of radii
y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

cqd_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
q, p = cqd_ivp['y'][0], cqd_ivp['y'][1]
dydt = [dynamics_fn(None, y) for y in cqd_ivp['y'].T]
dydt = np.stack(dydt).T
dqdt, dpdt = np.split(dydt,2)

# add noise
q += np.random.randn(*q.shape)*noise_std
p += np.random.randn(*p.shape)*noise_std

# %% Plot
fig1, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,6), dpi=100)
plt.subplots_adjust(wspace=0.3, hspace=0.25)
ax = ax.flatten()
ax[0].plot(q)
ax[0].set_ylabel('$q$'); ax[0].set_xlabel('Time (s)')
ax[1].plot(p)
ax[1].set_ylabel('$p$'); ax[1].set_xlabel('Time (s)')
ax[2].plot(dqdt[0])
ax[2].set_ylabel(''r'$\frac{dq}{dt}$'); ax[2].set_xlabel('Time (s)')
ax[3].plot(dpdt[0])
ax[3].set_ylabel(''r'$\frac{dp}{dt}$'); ax[3].set_xlabel('Time (s)')
