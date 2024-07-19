#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:42:19 2022

@author: user
"""

import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import inv

"""
Generalized stiffness matrix
"""
def kmat(k, dof):
    kk = np.zeros([2*dof, 2*dof])
    for i in range(2*dof):
        if i == 0:
            kk[i,1] = 1
        elif i%2 == 0:
            kk[i,i+1] = 1
        elif i == 1:
            kk[i,i-1] = -k
            kk[i,i+1] = k
        elif i == 2*dof-1:
            kk[i,i-1] = -k
            kk[i,i-3] = k
        else:
            kk[i,i+1] = k
            kk[i,i-1] = -2*k
            kk[i,i-3] = k
    return kk

def kmat_(k, dof):
    kk = np.zeros([dof, dof])
    for i in range(dof):
        if i == 0:
            kk[i,i] = (k[0]+k[1])
            kk[i,i+1] = -k[1]
        elif i == dof-1:
            kk[i,i] = k[-1]
            kk[i,i-1] = -k[-1]
        else:
            kk[i,i-1] = -k[i]
            kk[i,i] = (k[i]+k[i+1])
            kk[i,i+1] = -k[i+1]
    return kk


"""
The Response Generation Part: Linear system:
"""
def linear(x0, tparam, param):
    # Function for the dynamical systems:
    def F(t,x,params):
        k, m = params
        y = np.dot(np.array([[0, 1], [-k/m, 0]]), x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    k, m = param
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(param,))
    xt = np.vstack(sol.y)
    return xt, t_eval


"""
The Response Generation Part: Cubic-Quintic Duffing Oscillator:
"""
def duffing_quintic(x0, tparam, params):
    # Function for the dynamical systems:
    def F(t, x, params):
        a, b, c = params
        y = np.array([x[1], -a*x[0]-b*x[0]**3-c*x[0]**5])
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval= t_eval, args=(params,))
    xt = np.vstack(sol.y)    
    return xt, t_eval


"""
Atomic chain:
"""
def triatomic_molecule(x0, tparam, param):
    # Function for the dynamical systems:
    def F(t,x,params):
        m1, m2, k = params
        y = np.dot(np.array([[0,1,0,0,0,0],
                             [-k/m1,0,k/m1,0,0,0],
                             [0,0,0,1,0,0],
                             [k/m2,0,-2*k/m2,0,k/m2,0],
                             [0,0,0,0,0,1],
                             [0,0,k/m1,0,-k/m1,0]]), x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(param,))
    xt = np.vstack(sol.y)
    return xt, t_eval

def atomic_chain(x0, dof, tparam):
    # Function for the dynamical systems:
    def F(t,x,params):
        y = np.dot(params, x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    m1, m2, k = 1, 1, 1870
    kk = kmat(k, dof)
    kk[1::4] = kk[1::4]/m1
    kk[3::4] = kk[3::4]/m2
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(kk,))
    xt = np.vstack(sol.y)
    return xt, t_eval


"""
3-DOF system:
"""
def mdof_system(x0, tparam, param):
    # Function for the dynamical systems:
    def F(t,x,params):
        m1, m2, m3, k1, k2, k3 = params
        y = np.dot(np.array([[0,1,0,0,0,0],
                             [-(k1+k2)/m1,0,k2/m1,0,0,0],
                             [0,0,0,1,0,0],
                             [k2/m2,0,-(k2+k3)/m2,0,k3/m2,0],
                             [0,0,0,0,0,1],
                             [0,0,k3/m3,0,-k3/m3,0]]), x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(param,))
    xt = np.vstack(sol.y)
    return xt, t_eval

def mdof(x0, dof, param, tparam):
    # Function for the dynamical systems:
    def F(t,x,params):
        y = np.dot(params, x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    m = np.eye(dof)
    k = param*np.ones([dof])
    kk = kmat_(k, dof)
    mat = np.vstack(( np.hstack(( np.zeros([dof,dof]), np.eye(dof) )), \
                     np.hstack(( -np.dot(inv(m), kk), -np.zeros([dof,dof]) )) ))

    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(mat,))
    xt = np.vstack(sol.y)
    return xt, t_eval


"""
Rigid Body dynamics
"""
def rigid_body(x0, tparam):

    # Function for the dynamical systems:
    def F(t,x,params):
        I1, I2, I3 = params
        T1, T2, T3 = 0, 0, 0
        y = np.array([((I2-I3)/(I2*I3))*(x[1]*x[2]),
                      ((I3-I1)/(I1*I3))*(x[0]*x[2]),
                      ((I1-I2)/(I2*I1))*(x[0]*x[1])]) + \
            np.dot(np.eye(3), np.array([T1,T2,T3]))
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    I1, I2, I3 = 1, 1/2, 1/3
    params = [I1, I2, I3]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
    xt = np.vstack(sol.y)
    
    vel = np.zeros(xt.shape)
    for i in range(len(xt[0])):
        vel[:,i] = F(0,xt[:,i],params)
    return xt, vel, t_eval
