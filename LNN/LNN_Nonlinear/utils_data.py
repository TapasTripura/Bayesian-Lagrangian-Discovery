#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 16:32:29 2022

@author: user
"""

import numpy as np
import utils
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
def linear(x0, tparam):
    # Function for the dynamical systems:
    def F(t,x,params):
        k, m = params
        y = np.dot(np.array([[0, 1], [-k/m, 0]]), x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    k = 5000
    m = 1
    params = [k, m]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
    xt = np.vstack(sol.y)
    return xt, t_eval


"""
Atomic chain:
"""
def triatomic_molecule(x0, tparam):
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
    m1, m2, k = 1, 1, 1870
    params = [m1, m2, k]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
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
def mdof_system(x0, tparam):
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
    m1, m2, m3, k1, k2, k3 = 1, 1, 1, 5000, 5000, 5000
    params = [m1, m2, m3, k1, k2, k3]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
    xt = np.vstack(sol.y)
    return xt, t_eval

def mdof(x0, dof, tparam):
    # Function for the dynamical systems:
    def F(t,x,params):
        y = np.dot(params, x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    m = np.eye(dof)
    k = 5000*np.ones([dof])
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


"""
Code for string vibration in FINITE DIFFERENCE
"""
def string(L,stopTime,c,dx=0.01,dt=0.001):    
    # dx = 0.01   # Spacing of points on string
    # dt = 0.001  # Size of time step
    # c = 5  # Speed of wave propagation
    # L = 10 # Length of string
    # stopTime = 5 # Time to run the simulation
    
    r = c*dt/dx 
    n = int(L/dx + 1)
    t  = np.arange(0, stopTime+dt, dt)
    mesh = np.arange(0, L+dt, dx)
    sol = np.zeros([len(mesh), len(t)])
    
    # Set current and past to the graph of a plucked string
    current = 0.1 - 0.1*np.cos( 2*np.pi/L*mesh ) 
    past = current
    sol[:, 0] = current
    
    for i in range(len(t)):
        future = np.zeros(n)
    
        # Calculate the future position of the string
        future[0] = 0 
        future[1:n-2] = r**2*( current[0:n-3]+ current[2:n-1] ) + 2*(1-r**2)*current[1:n-2] - past[1:n-2]
        future[n-1] = 0 
        sol[:, i] = current
        
        # Settings up for the next time step
        past = current 
        current = future 
    
    Vel = np.zeros([sol.shape[0], sol.shape[1]])
    for i in range(1, sol.shape[1]-1):
        Vel[:,i] = (sol[:,i+1] - sol[:,i-1])/(2*dt)
    Vel[:,0] = (-3.0/2*sol[:,0] + 2*sol[:,1] - sol[:,2]/2) / dt
    Vel[:,sol.shape[1]-1] = (3.0/2*sol[:,sol.shape[1]-1] - 2*sol[:,sol.shape[1]-2] + sol[:,sol.shape[1]-3]/2) / dt
    
    xt = np.zeros([2*sol.shape[0],sol.shape[1]])
    xt[::2] = sol
    xt[1::2] = Vel
    return xt


"""
Codes for free vibration of a cantilever
"""
def cantilever(params,T,dt,Ne=100):
    # L = 1 # this code is for L=1m 
    rho, b, d, A, L, E, I = params
    c1 = 0#1e-3
    c2 = 0#1e-11
    xx = np.arange(1/Ne, 1+1/Ne, 1/Ne)
    
    [Ma, Ka, _, _] = beam3fun.Beam3(rho,A,E,I,L/Ne,Ne+1,'cantilever')
    # print(c1*np.sum(Ma),c2*np.sum(Ka))
    # g
    Ca = (c1*Ma + c2*Ka)
    F = lambda t : 0             # free vibration
    # % for forced vibration e.g. F = lambda t: np.sin(2*t)
    
    # % ------------------------------------------------
    # Lambda = 1.875104069/L
    # Lambda = 4.694091133/L
    # Lambda = 7.854757438/L
    Lambda = 10.99554073/L
    # Lambda = 14.13716839/L
    # Lambda = 17.27875953/L

    h1 = np.cosh(Lambda*xx) -np.cos(Lambda*xx) -(np.cos(Lambda*L)+np.cosh(Lambda*L)) \
        /(np.sin(Lambda*L)+np.sinh(Lambda*L))*(np.sinh(Lambda*xx)-np.sin(Lambda*xx))
    h2 = Lambda*(np.sinh(Lambda*xx)+np.sin(Lambda*xx))-(np.cos(Lambda*L)+np.cosh(Lambda*L)) \
        /(np.sin(Lambda*L)+np.sinh(Lambda*L))*(np.cosh(Lambda*xx)-np.cos(Lambda*xx))*Lambda
    
    D0 = np.zeros(2*Ne)
    D0[0::2] = h1
    D0[1::2] = h2
    V0 = np.zeros(2*Ne)
    # dt = 1e-3
    # T = 1
    D, V, A = beam3fun.Newmark(Ma,Ca,Ka,F,D0,V0,dt,T)
    
    # % -------------------------------------------------
    Dis = D[0::2]
    # Vel = V[0::2]
    Acc = A[0::2]
    
    Vel = np.zeros([Dis.shape[0], Dis.shape[1]])
    for i in range(1, Dis.shape[1]-1):
        Vel[:,i] = (Dis[:,i+1] - Dis[:,i-1])/(2*dt)
    Vel[:,0] = (-3.0/2*Dis[:,0] + 2*Dis[:,1] - Dis[:,2]/2) / dt
    Vel[:,Dis.shape[1]-1] = (3.0/2*Dis[:,Dis.shape[1]-1] - 2*Dis[:,Dis.shape[1]-2] + Dis[:,Dis.shape[1]-3]/2) / dt
    
    xt = np.zeros([2*Dis.shape[0],Dis.shape[1]])
    xt[::2] = Dis
    xt[1::2] = Vel
    
    return Dis, Vel, Acc


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
    # a, b, c = 1000,5000,10000
    # params = [a, b, c]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval= t_eval, args=(params,))
    xt = np.vstack(sol.y)    
    return xt, t_eval
