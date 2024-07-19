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
    wc, wz, b = 100, 10, 1
    q1, q2, q3, p1, p2, p3 = np.split(coords,6)
    H = 0.5*(p1**2 + p2**2 + p3**2) - 0.25*wz**2*(q1**2 + q2**2 + q3**2) \
        - 0.5*wc*((q1*p2) - (p1*q2)) + 0.5*b*(q2 - q1)**2 # 3DOF hamiltonian (linear oscillator)
    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dp1dt, dp2dt, dp3dt, dq1dt, dq2dt, dq3dt = np.split(dcoords,6)
    S = np.concatenate([dq1dt, dq2dt, dq3dt, -dp1dt, -dp2dt, -dp3dt], axis=-1)
    return S

t_span=[0,1]
timescale=1000
radius=None
y0=None
noise_std=0.0
t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
# get initial state
if y0 is None:
    y0 = np.random.uniform(1e-1, 1e-3, size=6)
if radius is None:
    radius = np.random.rand()*1e-3 + 1e-2 # sample a range of radii
y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

mdof_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
q1, q2, q3, p1, p2, p3 = mdof_ivp['y'][0], mdof_ivp['y'][1], mdof_ivp['y'][2], \
                            mdof_ivp['y'][3], mdof_ivp['y'][4], mdof_ivp['y'][5]
dydt = [dynamics_fn(None, y) for y in mdof_ivp['y'].T]
dydt = np.stack(dydt).T
dq1dt, dq2dt, dq3dt, dp1dt, dp2dt, dp3dt = np.split(dydt,6)

# add noise
q1 += np.random.randn(*q1.shape)*noise_std
q2 += np.random.randn(*q2.shape)*noise_std
q3 += np.random.randn(*q3.shape)*noise_std
p1 += np.random.randn(*p1.shape)*noise_std
p2 += np.random.randn(*p2.shape)*noise_std
p3 += np.random.randn(*p3.shape)*noise_std

# %% Plot
fig1, ax = plt.subplots(nrows=2, ncols=6, figsize=(10,6), dpi=100)
plt.subplots_adjust(wspace=0.5, hspace=0.45)
ax = ax.flatten()
ax[0].plot(q1)
ax[0].set_title('$q1$'); ax[0].set_xlabel('Time (s)')
ax[1].plot(q2)
ax[1].set_title('$q2$'); ax[1].set_xlabel('Time (s)')
ax[2].plot(q3)
ax[2].set_title('$q3$'); ax[2].set_xlabel('Time (s)')
ax[3].plot(p1)
ax[3].set_title('$p1$'); ax[3].set_xlabel('Time (s)')
ax[4].plot(p2)
ax[4].set_title('$p2$'); ax[0].set_xlabel('Time (s)')
ax[5].plot(p3)
ax[5].set_title('$p3$'); ax[1].set_xlabel('Time (s)')

ax[6].plot(dq1dt[0])
ax[6].set_title(''r'$\frac{dq1}{dt}$'); ax[2].set_xlabel('Time (s)')
ax[7].plot(dq2dt[0])
ax[7].set_title(''r'$\frac{dq2}{dt}$'); ax[3].set_xlabel('Time (s)')
ax[8].plot(dq3dt[0])
ax[8].set_title(''r'$\frac{dq3}{dt}$'); ax[2].set_xlabel('Time (s)')
ax[9].plot(dp1dt[0])
ax[9].set_title(''r'$\frac{dp1}{dt}$'); ax[3].set_xlabel('Time (s)')
ax[10].plot(dp2dt[0])
ax[10].set_title(''r'$\frac{dp2}{dt}$'); ax[2].set_xlabel('Time (s)')
ax[11].plot(dp3dt[0])
ax[11].set_title(''r'$\frac{dp3}{dt}$'); ax[3].set_xlabel('Time (s)')

