#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code contains the functions for data-generation of dynamical systems
"""

import numpy as np
import beam3fun
import utils
from scipy.integrate import solve_ivp
from numpy.linalg import inv

"""
Generalized stiffness matrix
"""
def kmat(k, dof):
    """
    Constructs the Stiffness matrix of shear storeys
    
    Parameters
    ----------
    k : stiffness at each degree of freedom. 
    dof : degree-of-freedom..

    Returns
    -------
    kk : stiffness matrix.
    """
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
    """
    Constructs the Stiffness matrix of molecule chain
    
    Parameters
    ----------
    k : stiffness of the molecule chains.
    dof : degree-of-freedom..

    Returns
    -------
    kk : stiffness matrix.
    """
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
The Response Generation Part: Two-Body system:
"""
def twobody(x0, tparam, G, m1, m2):
    """
    Simulates the Two-body problem
    
    Parameters
    ----------
    x0 : vector, initial condition.
    tparam : list, the temporal parameters.
    G : scalar, gravitational constant
    m1 : scalar, mass of first body
    m2 : scalar, mass of second body

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """    
    def F(t,z,G,m1,m2):
        dz = np.zeros(8)
        
        r = ((z[0] - z[4])**2 + (z[1] - z[5])**2)**(3/2)
        alpha = G/r
        
        dz[0] = z[2]
        dz[1] = z[3]
        dz[2] = alpha * m2 * (z[4] - z[0])
        dz[3] = alpha * m2 * (z[5] - z[1])
        dz[4] = z[6]
        dz[5] = z[7]
        dz[6] = alpha * m1 * (z[0] - z[4])
        dz[7] = alpha * m1 * (z[1] - z[5])
        return dz
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.linspace(0,T,int(T/dt))
    
    # Time integration:
    sol = solve_ivp(F, [0, T], x0, method='RK45', t_eval=t_eval, args=(G,m1,m2,),
                    rtol=1e-10, atol=1e-10)
    xt = sol.y
    return xt, t_eval


"""
The Response Generation Part: Linear system:
"""
def linear(x0, tparam):
    """
    Simulates the Linear spring-mass-dashpot system
    
    Parameters
    ----------
    x0 : vector, initial condition.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
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
Plane pendulum:
"""
def plane_pendulum(x0, tparam):
    """
    Simulates the plane pendulum
    
    Parameters
    ----------
    x0 : vector, initial condition.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
    def F(t,x,params):
        g, l = params
        y = np.array([x[1], -(g*np.sin(x[0]))/l])
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    g = 9.81
    l = 2
    params = [g, l]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
    xt = np.vstack(sol.y)
    return xt, t_eval


"""
Atomic chain:
"""
def triatomic_molecule(x0, tparam):
    """
    Simulates the Tri-atomic chain
    
    Parameters
    ----------
    x0 : vector, initial condition.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
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
    """
    Simulates the n-dimensional atomic chain
    
    Parameters
    ----------
    x0 : vector, initial condition.
    dof : integer, degree-of-freedom. 
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
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
    """
    Simulates the 3DOF system
    
    Parameters
    ----------
    x0 : vector, initial condition.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
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
    """
    Simulates an n-DOF system
    
    Parameters
    ----------
    x0 : vector, initial condition.
    dof : integer, degree-of-freedom.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
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
def penning_trap(x0, params, tparam):
    """
    Simulates the Penning-Trap
    
    Parameters
    ----------
    x0 : vector, initial condition.
    params : list, the system parameters.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
    # Function for the dynamical systems:
    def F(t,x,params):
        wc, wz = params
        y = np.dot(np.array([[0,1,0,0,0,0],
                             [0.5*wz**2,0,0,wc,0,0],
                             [0,0,0,1,0,0],
                             [0,-wc,0.5*wz**2,0,0,0],
                             [0,0,0,0,0,1],
                             [0,0,0,0,-wz**2,0]]), x)
        return y
    
    # The time parameters:
    dt, t0, T = tparam
    t_eval = np.arange(0, T, dt)
    wc, wz = params
    params = [wc, wz]
    
    # Time integration:
    sol = solve_ivp(F, [t0, T], x0, method='RK45', t_eval=t_eval, args=(params,))
    xt = np.vstack(sol.y)
    return xt, t_eval


"""
Code for string vibration in FINITE DIFFERENCE
"""
def string(L,stopTime,c,dx=0.01,dt=0.001):    
    """
    Parameters
    ----------
    L : scalar, length of string.
    stopTime : scalar, final time.
    c : float, the velocity of wave.
    dx : float, spatial discretization step.
    dt : float, time step.

    Returns
    -------
    xt : matrix [2 x nx x nt], states containing [u(x,t), \dot{u}(x,t)].
    """
    
    r = c*dt/dx 
    n = int(L/dx + 1)
    t  = np.arange(0, stopTime+dt, dt)
    mesh = np.arange(0, L+dx, dx)
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
    """
    Parameters
    ----------
    params : list, the system parameters.
    T : scalar, terminal time.
    dt : float, time step.
    Ne : integer, numver of finite element. The default is 100.

    Returns
    -------
    Dis : matrix, displacement.
    Vel : matrix, velocity.
    Acc : matrix, acceleration.
    """
    # L = 1 # this code is for L=1m 
    rho, b, d, A, L, E, I = params
    c1 = 0 
    c2 = 0 
    xx = np.arange(1/Ne, 1+1/Ne, 1/Ne)
    
    [Ma, Ka, _, _] = beam3fun.Beam3(rho,A,E,I,L/Ne,Ne+1,'cantilever')
    
    Ca = (c1*Ma + c2*Ka)
    F = lambda t : 0             # free vibration
    # for forced vibration e.g. F = lambda t: np.sin(2*t)
    
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
    """
    Simulates the Penning-Trap
    
    Parameters
    ----------
    x0 : vector, initial condition.
    params : list, the system parameters.
    tparam : list, the temporal parameters.

    Returns
    -------
    xt : solution matrix.
    t_eval : The time vector.
    """
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
Code for channel flow using Navier-Stokes equation
    - Note the presence of the pseudo-time variable nit. 
    - This sub-iteration in the Poisson calculation helps ensure a divergence-free field.
"""
nit = 100

def build_up_b_channel(rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    
    # Periodic BC Pressure @ x = 2
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                          2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                               (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                         2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                              (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                         ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))
    return b

def pressure_poisson_periodic_channel(p, dx, dy, b):
    pn = np.zeros_like(p)
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        # Periodic BC Pressure @ x = 0
        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                       (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])
        
        # Wall boundary conditions, pressure
        p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
    
    return p

def channel_flow(nt, u, v, dt, dx, dy, p, rho, nu, F):   
    """
    Simulates the channel flow.

    Parameters
    ----------
    nt : float, total number of time step.
    u : matrix, initial displacement field.
    v : matrix, initial velocity field.
    dt : float, time step.
    dx : float, spatial discretization step in x-dir.
    dy : float, spatial discretization step in y-dir.
    p : matrix, initial pressure field.
    rho : scalar, fluid density.
    nu : scalar, dynamic viscosity.

    Returns
    -------
    ut : matrix, displacement field.
    vt : matrix, velocity field.
    pt : matrix, pressure field.
    
    """
    ut = np.zeros((u.shape[0], u.shape[1], nt))
    vt = np.zeros((v.shape[0], v.shape[1], nt))
    pt = np.zeros((p.shape[0], p.shape[1], nt))

    for n in range(nt):
        un = u
        vn = v

        b = build_up_b_channel(rho, dt, dx, dy, u, v)
        p = pressure_poisson_periodic_channel(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                               dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + 
                         F[1:-1, 1:-1] * dt)
    
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                               dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
    
        # Periodic BC u @ x = 2     
        u[1:-1, -1] = (un[1:-1, -1] - 
                       un[1:-1, -1] * dt / dx * (un[1:-1, -1] - un[1:-1, -2]) -
                       vn[1:-1, -1] * dt / dy * (un[1:-1, -1] - un[0:-2, -1]) -
                       dt / (2 * rho * dx) * (p[1:-1, 0] - p[1:-1, -2]) + 
                       nu * (dt / dx**2 * (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                       dt / dy**2 * (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F[1:-1, -1] * dt)
    
        # Periodic BC u @ x = 0
        u[1:-1, 0] = (un[1:-1, 0] - 
                      un[1:-1, 0] * dt / dx * (un[1:-1, 0] - un[1:-1, -1]) -
                      vn[1:-1, 0] * dt / dy * (un[1:-1, 0] - un[0:-2, 0]) - 
                      dt / (2 * rho * dx) * (p[1:-1, 1] - p[1:-1, -1]) + 
                      nu * (dt / dx**2 * (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                      dt / dy**2 * (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F[1:-1, 0] * dt)
    
        # Periodic BC v @ x = 2
        v[1:-1, -1] = (vn[1:-1, -1] - 
                       un[1:-1, -1] * dt / dx * (vn[1:-1, -1] - vn[1:-1, -2]) - 
                       vn[1:-1, -1] * dt / dy * (vn[1:-1, -1] - vn[0:-2, -1]) -
                       dt / (2 * rho * dy) * (p[2:, -1] - p[0:-2, -1]) +
                       nu * (dt / dx**2 * (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                       dt / dy**2 * (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))
    
        # Periodic BC v @ x = 0
        v[1:-1, 0] = (vn[1:-1, 0] - 
                      un[1:-1, 0] * dt / dx * (vn[1:-1, 0] - vn[1:-1, -1]) -
                      vn[1:-1, 0] * dt / dy * (vn[1:-1, 0] - vn[0:-2, 0]) -
                      dt / (2 * rho * dy) * (p[2:, 0] - p[0:-2, 0]) +
                      nu * (dt / dx**2 * (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                      dt / dy**2 * (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))
    
        # Wall BC: u,v = 0 @ y = 0,2
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :]= 0
        
        ut[..., n] = u
        vt[..., n] = v
        pt[..., n] = p
    
    return ut, vt, pt


"""
Code for Cavity flow using Navier-Stokes equation
    - Note the presence of the pseudo-time variable nit. 
    - This sub-iteration in the Poisson calculation helps ensure a divergence-free field.
"""
nit = 50

def build_up_b(rho, dt, u, v, dx, dy):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.zeros_like(p)
    pn = p.copy()
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])

        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0        # p = 0 at y = 2
        
    return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    """
    Simulates the cavity flow.

    Parameters
    ----------
    nt : float, total number of time step.
    u : matrix, initial displacement field.
    v : matrix, initial velocity field.
    dt : float, time step.
    dx : float, spatial discretization step in x-dir.
    dy : float, spatial discretization step in y-dir.
    p : matrix, initial pressure field.
    rho : scalar, fluid density.
    nu : scalar, dynamic viscosity.

    Returns
    -------
    ut : matrix, displacement field.
    vt : matrix, velocity field.
    pt : matrix, pressure field.
    
    """
    ut = np.zeros((u.shape[0], u.shape[1], nt))
    vt = np.zeros((v.shape[0], v.shape[1], nt))
    pt = np.zeros((p.shape[0], p.shape[1], nt))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * (dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt / dy**2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                       (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                       (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                       (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                       (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
        
        ut[..., n] = u
        vt[..., n] = v
        pt[..., n] = p
        
    return ut, vt, pt
