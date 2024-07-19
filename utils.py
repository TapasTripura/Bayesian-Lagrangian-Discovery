#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code contains the useful functions for Lagrange discovery.
   The code is associated with...
-- A Bayesian framework for discovering interpretable Lagrangian of dynamical systems from data 
-- Authored by: Tapas Tripura and Souvik Chakraborty
"""

import sympy as sym
import numpy as np
from scipy import linalg as LA
from sklearn.metrics import mean_squared_error as MSE
from numpy import linalg as LA2
from numpy.random import gamma as IG
from numpy.random import beta
from numpy.random import binomial as bern
from numpy.random import multivariate_normal as mvrv
from scipy.special import loggamma as LG
import matplotlib.pyplot as plt

from timeit import default_timer


"""
Sparse-Least square regression
"""
def sparsifyDynamics(library,target,lam,iteration=10):
    """
    It performs the least-squares sparse-regression. 
    
    Parameters
    ----------
    library : matrix, the design matrix of candidate basis functions.
    target : vector, the target vector.
    lam : scalar, the sparsification constant.
    iteration : integer, number of sequential threshold iterations.

    Returns
    -------
    Xi : vector, the sparse parameter vector.
    """
    Xi = np.matmul(np.linalg.pinv(library), target.T) # initial guess: Least-squares
    for k in range(iteration):
        smallinds = np.where(abs(Xi) < lam)   # find small coefficients
        Xi[smallinds] = 0
        for ind in range(Xi.shape[1]):
            biginds = np.where(abs(Xi[:,ind]) > lam)
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.matmul(np.linalg.pinv(library[:, biginds[0]]), target[ind, :].T) 
    return Xi


"""
Euler-Lagrangian operator on library
"""
def euler_lagrange_mdof(D,xvar,xt,dt):
    """
    It obtains the Euler-Lagrange library by evaluating the Euler-Lagrange 
    operator on the actual Lagrange library matrix, 
    And, provides the target vector by removing the squared velocity terms
    from the design matrix, as explained in the paper.
    
    Parameters
    ----------
    D : matrix, the Lagrange matrix.
    xvar : symbolic vector, the variables.
    xt : vector, the numerical responses at the variables 'xvar'.
    dt : scalar, the time step.

    Returns
    -------
    Rl : matrix, the Euler-Lagrange library obtained after removing ith squared 
    velocity terms.
    Dydt : matrix, the collection of the squared velocity terms from 
    Euler-Lagrange library.
    """
    # Derivative with respect to displacment (change in potential energy):
    Drx = []
    for kk in range(len(xvar)):
        fun = [sym.diff(D[i], xvar[kk]) for i in range(len(D))]
        Drx.append(fun)
    
    diff_Drx = []
    for kk in range(len(xvar)):
        fun = sym.lambdify([xvar], Drx[kk], 'numpy')
        diff_Drx.append(fun)
        
    # Compute the numerial library:
    Dxdx = []
    for j in range(len(xvar)):
        Dxtemp = np.zeros([xt.shape[1], D.shape[0]])
        for i in range(xt.shape[1]):
            Dxtemp[i,:] = diff_Drx[j](xt[0:len(xvar),i])
        Dxdx.append(Dxtemp)
    
    momentum_index = xvar[1::2] # every second component
    momentum_library = Dxdx[1::2]    
    Dydt = []
    nd = len(D)
    for j in range(len(momentum_index)):
        Dydt_temp = np.zeros((xt.shape[1], nd))
        for i in range(nd):
            Dydt_temp[:,i] = FiniteDiff(momentum_library[j][:,i],dt,1)
        Dydt.append(Dydt_temp)
        
    Dxdx = Dxdx[0::2]    
    Rl = []
    for i in range(len(momentum_index)):
        Rl.append(Dydt[i] + Dxdx[i][:,:])
    
    dxdt = np.zeros([len(momentum_index),xt.shape[1]])
    for i in range(len(momentum_index)):
        dxdt[i,:] = Rl[i][:, np.where(D == momentum_index[i]**2)].squeeze()
     
    for i in range(len(momentum_index)):
        Rl[i] = np.delete(Rl[i], np.where(D == momentum_index[i]**2), axis=1)
    
    return Rl, dxdt


"""
The Dictionary creation part:
"""
def library_sdof(xt,polyn=2,harmonic=False,harmonic_interaction=False):
    """
    Obtains the Lagrangian library from the system responses (for SDOF ODEs)
    
    Parameters
    ----------
    xt : vector, the numerical responses.
    polyn : scalar, the polynomial order.
    harmonic : boolean function, if 1 adds harmonic functions in the library.
    harmonic_interaction : boolean, if 1 adds basis functions incorporating
                            interaction between states.

    Returns
    -------
    D : matrix, the design matrix Lagrangian.
    ind : scalar, the dimension of the candidate basis functions.
    """
    # Powers of states
    xt_disp = xt[::2]
    xt_vel = xt[1::2]
    # poly order 0
    D = 1            
    # square
    for i in range(len(xt_vel)):
        new = xt_vel[i]**2
        D = np.append(D, new)
    #
    for i in range(1,polyn+1):
        for j in range(len(xt_disp)):
            new = xt_disp[j]**i
            D = np.append(D, new)
    # Second order interaction between states
    for i in range(len(xt)):
        for j in  range(i+1,len(xt)):
            new = np.multiply(xt[i], xt[j])
            D = np.append(D, new)
            
    # harmonic components:
    if harmonic != False:
        # for sin(x)
        for i in range(len(xt)):
            new = sym.sin(xt[i])
            D = np.append(D, new)  
        # for cos(x)
        for i in range(len(xt)):
            new = sym.cos(xt[i])
            D = np.append(D, new)
            
    if harmonic_interaction != False:
        # for xsin(x)
        for i in range(len(xt)):
            for j in range(i,len(xt)):
                new = xt[i]*sym.sin(xt[j])
                D = np.append(D, new)
        # for xcos(x)
        for i in range(len(xt)):
            for j in range(i,len(xt)):
                new = xt[i]*sym.cos(xt[j])
                D = np.append(D, new)
        # for sin(x)cos(x)
        for i in range(len(xt)):
            for j in range(i,len(xt)):
                new = sym.sin(xt[i])*sym.cos(xt[j])
                D = np.append(D, new)
        
    ind = len(D)
    return D, ind

def library_mdof(xt,polyn,harmonic=False,harmonic_interaction=False):
    """
    Obtains the Lagrangian library from the system responses (for MDOF ODEs)
    
    Parameters
    ----------
    xt : vector, the numerical responses.
    polyn : scalar, the polynomial order.
    harmonic : boolean function, if 1 adds harmonic functions in the library.
    harmonic_interaction : boolean, if 1 adds basis functions incorporating
                            interaction between states.

    Returns
    -------
    D : matrix, the design matrix Lagrangian.
    ind : scalar, the dimension of the candidate basis functions.
    """
    # The displacement components
    xt_disp = xt[::2]
    
    # poly order 0
    D = 1            
    # square
    for i in range(1,polyn+1):
        for j in range(len(xt)):
            new = xt[j]**i
            D = np.append(D, new)
    if polyn >= 1:
        # difference -- poly order 1
        for i in range(len(xt_disp)):
            for j in range(i+1, len(xt_disp)):
                new = xt_disp[j]-xt_disp[i]
                D = np.append(D, new)
    if polyn >= 2:
        # difference -- poly order 2
        for i in range(len(xt_disp)):
            for j in range(i+1, len(xt_disp)):
                new = (xt_disp[j]-xt_disp[i])**2
                D = np.append(D, new)
    if polyn >= 3:
        # difference -- poly order 3
        for i in range(len(xt_disp)):
            for j in range(i+1, len(xt_disp)):
                new = (xt_disp[j]-xt_disp[i])**3
                D = np.append(D, new)
    if polyn >= 4:
        # difference -- poly order 4
        for i in range(len(xt_disp)):
            for j in range(i+1, len(xt_disp)):
                new = (xt_disp[j]-xt_disp[i])**4
                D = np.append(D, new)
    if polyn >= 5:
        # difference -- poly order 5
        for i in range(len(xt_disp)):
            for j in range(i+1, len(xt_disp)):
                new = (xt_disp[j]-xt_disp[i])**5
                D = np.append(D, new)
                
    # harmonic components:
    if harmonic != False:
        # for sin(x)
        for i in range(len(xt)):
            new = sym.sin(xt[i])
            D = np.append(D, new)      
        # for cos(x)
        for i in range(len(xt)):
            new = sym.cos(xt[i])
            D = np.append(D, new)
        # for sin(x-x')
        for i in range(len(xt)-1):
            new = sym.sin(xt[i+1] - xt[i])
            D = np.append(D, new)      
        # for cos(x-x')
        for i in range(len(xt)-1):
            new = sym.cos(xt[i+1] - xt[i])
            D = np.append(D, new)

    if harmonic_interaction != False:
        # for xsin(x)
        for i in range(len(xt)):
            for j in range(i,len(xt)):
                new = xt[i]*sym.sin(xt[j])
                D = np.append(D, new)
        # for xcos(x)
        for i in range(len(xt)):
            for j in range(i,len(xt)):
                new = xt[i]*sym.cos(xt[j])
                D = np.append(D, new)
        # for sin(x)cos(x)
        for i in range(len(xt)):
            for j in range(i,len(xt)):
                new = sym.sin(xt[i])*sym.cos(xt[j])
                D = np.append(D, new)
    ind = len(D)
    return D, ind


def library_pde(xt,Type,dx,polyn=0,harmonic=0):
    """
    Obtains the Lagrangian library from the system responses (for PDEs)
    
    Parameters
    ----------
    xt : vector, the numerical responses.
    Type : string, the order of PDE.
           order1 = first or second order systems, like wave equation 
           order1 = fourth-order systems, like Euler-Bernoulli equation
    dx : scalar, the grid spacing.
    polyn : scalar, the polynomial order.

    Returns
    -------
    D : matrix, the design matrix Lagrangian.
    ind : scalar, the dimension of the candidate basis functions.
    """
    # The polynomial is (x1 + x2)^p, with p is the order
    xt_disp = xt[::2]
    xt_vel = xt[1::2]
    D = 1
    # square (This library function must be present)
    for i in range(len(xt_vel)):
        new = xt_vel[i]**2
        D = np.append(D, new) 
        
    if Type == 'order1':   
        # difference -- poly order 1
        D = np.append(D, xt_disp[0]/dx)
        for i in range(1,len(xt_disp)-1):
            new = (xt_disp[i+1] - xt_disp[i])/dx
            D = np.append(D, new)
        D = np.append(D, -1*xt_disp[-1]/dx)
    
        if polyn >= 2:
            # difference -- poly order 2
            D = np.append(D, xt_disp[0]**2/(dx**2))
            for i in range(len(xt_disp)-1):
                new = (xt_disp[i+1] - xt_disp[i])**2/(dx**2)
                D = np.append(D, new)
            D = np.append(D, xt_disp[-1]**2/(dx**2))
            
        if polyn >= 3:
            # difference -- poly order 3
            D = np.append(D, xt_disp[0]**3/(dx**3))
            for i in range(len(xt_disp)-1):
                new = (xt_disp[i+1] - xt_disp[i])**3/(dx**3)
                D = np.append(D, new)
            D = np.append(D, xt_disp[-1]**3/(dx**3))
            
        if polyn >= 4:
            # difference -- poly order 4
            D = np.append(D, xt_disp[0]**4/(dx**4))
            for i in range(len(xt_disp)-1):
                new = (xt_disp[i+1] - xt_disp[i])**4/(dx**4)
                D = np.append(D, new)
            D = np.append(D, xt_disp[-1]**4/(dx**4))
            
    if Type == 'order2':

        if polyn >= 2:
            # difference -- poly order 2
            D = np.append(D, xt_disp[0]**2/dx**4)
            D = np.append(D, (xt_disp[1] - 2*xt_disp[0])**2/(dx**4))
            for i in range(1, len(xt_disp)-1):
                new = (xt_disp[i+1] - 2*xt_disp[i] + xt_disp[i-1])**2/(dx**4)
                D = np.append(D, new)
            D = np.append(D, (-2*xt_disp[-1] + xt_disp[-2])**2/(dx**4))
            D = np.append(D, xt_disp[-1]**2/dx**4)
            
        if polyn >= 3:
            # difference -- poly order 2
            D = np.append(D, (xt_disp[1] - 2*xt_disp[0])**3/(dx**6))
            for i in range(1, len(xt_disp)-1):
                new = (xt_disp[i+1] - 2*xt_disp[i] + xt_disp[i-1])**3/(dx**6)
                D = np.append(D, new)
            D = np.append(D, (-2*xt_disp[-1] + xt_disp[-2])**3/(dx**6))
            
        if polyn >= 4:
            # difference -- poly order 2
            D = np.append(D, (xt_disp[1] - 2*xt_disp[0])**4/(dx**8))
            for i in range(1, len(xt_disp)-1):
                new = (xt_disp[i+1] - 2*xt_disp[i] + xt_disp[i-1])**4/(dx**8)
                D = np.append(D, new)
            D = np.append(D, (-2*xt_disp[-1] + xt_disp[-2])**4/(dx**8))
    
    if Type == 'order3':
        D = np.append(D, (xt_disp[2]-2*xt_disp[1])/(dx**3))
        D = np.append(D, (xt_disp[3]-2*xt_disp[2]+2*xt_disp[0])/(dx**3))
        for i in range(2,len(xt_disp)-2):
            new = (xt_disp[i+2]-2*xt_disp[i+1]+2*xt_disp[i-1]-xt_disp[i-2])/(dx**3)
            D = np.append(D, new)
        D = np.append(D, (-2*xt_disp[-1]+2*xt_disp[-3]-xt_disp[-4])/(dx**3))
        D = np.append(D, (2*xt_disp[-2]-xt_disp[-3])/(dx**3))
        
        if polyn >= 2:
            # difference -- poly order 3
            D = np.append(D, (xt_disp[2]-2*xt_disp[1])**2/(dx**6))
            D = np.append(D, (xt_disp[3]-2*xt_disp[2]+2*xt_disp[0])**2/(dx**6))
            for i in range(2,len(xt_disp)-2):
                new = (xt_disp[i+2]-2*xt_disp[i+1]+2*xt_disp[i-1]-xt_disp[i-2])**2/(dx**6)
                D = np.append(D, new)
            D = np.append(D, (-2*xt_disp[-1]+2*xt_disp[-3]-xt_disp[-4])**2/(dx**6))
            D = np.append(D, (2*xt_disp[-2]-xt_disp[-3])**2/(dx**6))
            
        if polyn >= 3:
            D = np.append(D, (xt_disp[2]-2*xt_disp[1])**3/(dx**9))
            D = np.append(D, (xt_disp[3]-2*xt_disp[2]+2*xt_disp[0])**3/(dx**9))
            for i in range(2,len(xt_disp)-2):
                new = (xt_disp[i+2]-2*xt_disp[i+1]+2*xt_disp[i-1]-xt_disp[i-2])**3/(dx**9)
                D = np.append(D, new)
            D = np.append(D, (-2*xt_disp[-1]+2*xt_disp[-3]-xt_disp[-4])**3/(dx**9))
            D = np.append(D, (2*xt_disp[-2]-xt_disp[-3])**3/(dx**9))
    
    if Type == 'order4':
        D = np.append(D, (xt_disp[2]-4*xt_disp[1]+6*xt_disp[0])/(dx**4))
        D = np.append(D, (xt_disp[3]-4*xt_disp[2]+6*xt_disp[1]-4*xt_disp[0])/(dx**4))
        for i in range(2,len(xt_disp)-2):
            new = (xt_disp[i+2]-4*xt_disp[i+1]+6*xt_disp[i]-4*xt_disp[i-1]-xt_disp[i-2])/(dx**4)
            D = np.append(D, new)
        D = np.append(D, (-4*xt_disp[-1]+6*xt_disp[-2]-4*xt_disp[-3]-xt_disp[-4])/(dx**4))
        D = np.append(D, (6*xt_disp[-1]-4*xt_disp[-2]-xt_disp[i-3])/(dx**4))
        
        if polyn >=2:
            D = np.append(D, (xt_disp[2]-4*xt_disp[1]+6*xt_disp[0])**2/(dx**8))
            D = np.append(D, (xt_disp[3]-4*xt_disp[2]+6*xt_disp[1]-4*xt_disp[0])**2/(dx**8))
            for i in range(2,len(xt_disp)-2):
                new = (xt_disp[i+2]-4*xt_disp[i+1]+6*xt_disp[i]-4*xt_disp[i-1]-xt_disp[i-2])**2/(dx**8)
                D = np.append(D, new)
            D = np.append(D, (-4*xt_disp[-1]+6*xt_disp[-2]-4*xt_disp[-3]-xt_disp[-4])**2/(dx**8))
            D = np.append(D, (6*xt_disp[-1]-4*xt_disp[-2]-xt_disp[i-3])**2/(dx**8))
        
        if polyn >= 3:
            D = np.append(D, (xt_disp[2]-4*xt_disp[1]+6*xt_disp[0])**3/(dx**12))
            D = np.append(D, (xt_disp[3]-4*xt_disp[2]+6*xt_disp[1]-4*xt_disp[0])**3/(dx**12))
            for i in range(2,len(xt_disp)-2):
                new = (xt_disp[i+2]-4*xt_disp[i+1]+6*xt_disp[i]-4*xt_disp[i-1]-xt_disp[i-2])**3/(dx**12)
                D = np.append(D, new)
            D = np.append(D, (-4*xt_disp[-1]+6*xt_disp[-2]-4*xt_disp[-3]-xt_disp[-4])**3/(dx**12))
            D = np.append(D, (6*xt_disp[-1]-4*xt_disp[-2]-xt_disp[i-3])**3/(dx**12))
    
    # harmonic components:
    if harmonic == 1:
        # for sin(x)
        for i in range(len(xt_disp)):
            new = sym.sin(xt_disp[i])
            D = np.append(D, new)      
        # for cos(x)
        for i in range(len(xt_disp)):
            new = sym.cos(xt_disp[i])
            D = np.append(D, new)
            
    ind = len(D)
    return D, ind


"""
For numerical derivative using 4th order accuarcy
"""
def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3
    
    Parameters
    ----------
    u : vector, data to be differentiated
    dx : scalar, Grid spacing.  Assumes uniform spacing
    d : order of derivative
    
    Returns
    -------
    ux : vector, the derivative vector
    """
    
    n = u.size
    ux = np.zeros(n)
    
    if d == 1:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-u[i-1]) / (2*dx)
        
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux
    
    if d == 2:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2
        
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux
    
    if d == 3:
        for i in range(2,n-2):
            ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3
        
        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux
    
    if d > 3:
        return FiniteDiff(FiniteDiff(u,dx,3), dx, d-3)


def FiniteDiffUnequal(u,x,d):
    """
    Takes dth derivative data using Lagrange polynomial (up to d=2)
    Works but with poor accuracy for d > 2
    
    Parameters
    ----------
    u : array, function values.
    x : array, the coordinate on which function derivative is required.
    d : scalar, order of the derivative, upto 2 only.

    Returns
    -------
    ux : array, derivative with equal/unequal intervals.
    """
    n = len(u)
    ux = np.zeros(n)
    if d == 1:
        for i in range(1,n-1):
            h1 = np.abs(x[i] - x[i-1])
            h2 = np.abs(x[i+1] - x[i])
            ux[i] = - (h2*u[i-1])/(h1*(h1+h2)) - ((h1-h2)*u[i])/(h1*h2) \
                    + (h1*u[i+1])/(h2*(h1+h2)) 
        
        h10 = np.abs(x[1] - x[0])
        h20 = np.abs(x[2] - x[1])
        ux[0] = - ((2*h10+h20)*u[0])/(h10*(h10+h20)) + ((h10+h20)*u[1])/(h10*h20) \
                - (h10*u[2])/(h20*(h10+h20)) 
        
        h1n = np.abs(x[n-2] - x[n-3])
        h2n = np.abs(x[n-1] - x[n-2])
        ux[n-1] = (h2n*u[i-3])/(h1n*(h1n+h2n)) - ((h1n+h2n)*u[i-2])/(h1n*h2n) \
                + ((h1n+2*h2n)*u[i-1])/(h2n*(h1n+h2n)) 
        return ux
    
    if d == 2:
        for i in range(1,n-1):
            h1 = np.abs(x[i] - x[i-1])
            h2 = np.abs(x[i+1] - x[i])
            ux[i] = 2*(h2*u[i-1] - (h1+h2)*u[i] + h1*u[i+1]) / (h1*h2*(h1+h2)) 
        
        h10 = np.abs(x[1] - x[0])
        h20 = np.abs(x[-2] - x[-1])
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / h10**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / h20**2
        return ux
        
    if d > 2:
        return FiniteDiffUnequal(FiniteDiffUnequal(u,x,2), x, d-2)
    
    
def SpectralDiff(u,L):
    """
    Approximate derivative by FFT
    
    Parameters
    ----------
    u : values of some function
    L : Range of the function
    
    Returns
    -------
    ux : vector, the derivative 
    """
    n = len(u)
    fhat = np.fft.fft(u)
    kappa = (2*np.pi/L)*np.arange(-n/2,n/2)
    # Re-order fft frequencies
    kappa = np.fft.fftshift(kappa) 
    # Obtain real part of the function for plotting
    ux = kappa*fhat*(1j)
    # Inverse Fourier Transform
    ux = np.real(np.fft.ifft(ux))
    return ux



def PolyDiff(u, x, deg = 3, diff = 1, width = 5):
    """
    It evaluates the derivative using polynomial fit
    
    Parameters
    ----------
    u : vector, values of some function
    x : vector, x-coordinates where values are known
    deg : integer, degree of polynomial to use
    diff : integer, maximum order derivative we want
    width : integer, width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    
    Returns
    -------
    du : vector, the derivative 
    See https://github.com/snagcliffs/PDE-FIND/blob/master/PDE_FIND.py
    """
    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2*width,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        # Note code originally used an even number of points here.
        # This is an oversight in the original code fixed in 2022.
        points = np.arange(j - width, j + width + 1)

        # Fit to a polynomial
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            du[j-width, d-1] = poly.deriv(m=d)(x[j])

    return du


"""
Theta: Multivariate Normal distribution
"""
def sigmu(z, D, vs, xdts, eps=0.00000001):
    """
    It provides the posterior mean and standard deviation 
    
    Parameters
    ----------
    z : array, latent indicator variable vector.
    D : matrix, the design matrix of library functions.
    vs : scalar, slab variance.
    xdts : array, the label vector.

    Returns
    -------
    mu : vector, the posterior mean of theta. 
    BSIG : matrix, the posterior covariance of theta. 
    Aor : matrix, slab covariance. 
    index : vector, locations of activate basses. 
    """
    
    index = np.array(np.where(z != 0))
    index = np.reshape(index,-1) # converting to 1-D array, 
    Dr = D[:,index] 
    Aor = np.eye(len(index)) # independent prior
    # Aor = np.dot(len(Dr), LA2.inv(np.matmul(Dr.T, Dr) + eps)) # g-prior
    BSIG = LA2.inv(np.matmul(Dr.T,Dr) + np.dot(pow(vs,-1), LA2.inv(Aor)) + eps)
    mu = np.matmul(np.matmul(BSIG,Dr.T),xdts)
    return mu, BSIG, Aor, index

"""
P(Y|zi=(0|1),z-i,vs)
"""
def pyzv(D, ztemp, vs, N, xdts, asig, bsig, eps=0.00000001):
    """
    It provides the posterior mean and standard deviation 
    
    Parameters
    ----------
    D : matrix, the design matrix of library functions.
    ztemp : array, latent indicator variable vector.
    vs : scalar, slab variance.
    N : integer, number of time points.
    xdts : array, the label vector.
    asig, bsig : scalars, the hyperparameters.

    Returns
    -------
    PZ1 : scalar, the probability of getting selected. 
    """
    rind = np.array(np.where(ztemp != 0))[0]
    rind = np.reshape(rind, -1) # converting to 1-D array,   
    Sz = sum(ztemp)
    Dr = D[:, rind] 
    Aor = np.eye(len(rind)) # independent prior
    # Aor = np.dot(N, LA2.inv(np.matmul(Dr.T, Dr) + eps)) # g-prior
    BSIG = np.matmul(Dr.T, Dr) + np.dot(pow(vs, -1),LA2.inv(Aor))
    
    (sign, logdet0) = LA2.slogdet(LA2.inv(Aor + eps))
    (sign, logdet1) = LA2.slogdet(LA2.inv(BSIG + eps))
    
    PZ = LG(asig + 0.5*N) -0.5*N*np.log(2*np.pi) - 0.5*Sz*np.log(vs) \
        + asig*np.log(bsig) - LG(asig) + 0.5*logdet0 + 0.5*logdet1
    denom1 = np.eye(N) - np.matmul(np.matmul(Dr, LA2.inv(BSIG + eps)), Dr.T)
    denom = (0.5*np.matmul(np.matmul(xdts.T, denom1), xdts))
    PZ = PZ - (asig+0.5*N)*(np.log(bsig + denom))
    if np.isnan(PZ) == 1: 
        PZ = 0
    return PZ

"""
P(Y|zi=0,z-i,vs)
"""
def pyzv0(xdts, N, asig, bsig):
    """
    It provides the posterior mean and standard deviation 
    
    Parameters
    ----------
    xdts : array, the label vector.
    N : integer, number of time points.
    asig, bsig : scalars, the hyperparameters.

    Returns
    -------
    PZ0 : scalar, the probability of not getting selected. 
    """
    PZ0 = LG(asig + 0.5*N) - 0.5*N*np.log(2*np.pi) + asig*np.log(bsig) - LG(asig) \
        + np.log(1) - (asig+0.5*N)*np.log(bsig + 0.5*np.matmul(xdts.T, xdts))
    return PZ0


"""
Sparse regression with Normal Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
def sparse(xdts, D, nl, MCMC, burn_in, eps=0.00000001):    
    """
    Function for performing MCMC using Gibbs sampler.
    
    Parameters
    ----------
    xdts : array, the label vector.
    D : matrix, the design matrix of library functions.
    nl : scalar, total number of library basses.
    MCMC : scalar, total length of Markov chain.
    burn_in : scalar, number of burn-in samples to discard.
    """
    
    # Residual variance:
    err_var = res_var(D, xdts)

    """
    # Gibbs sampling:
    """
    # Hyper-parameters
    ap, bp = 0.1, 1 # for beta prior for p0
    av, bv = 0.5, 0.5 # inverge gamma for vs
    asig, bsig = 1e-2, 1e-2 # invese gamma for sig^2

    # Parameter Initialisation:
    p0 = np.zeros(MCMC)
    vs = np.zeros(MCMC)
    sig = np.zeros(MCMC)
    p0[0] = 0.1
    vs[0] = 10
    sig[0] = err_var

    N = len(xdts)

    # Initial latent vector
    zval = np.zeros(nl)
    zint  = latent_fb_dual(nl, D, xdts)
    zstore = np.transpose(np.vstack([zint]))
    zval = zint

    zval0 = zval
    vs0 = vs[0]
    mu, BSIG, Aor, index = sigmu(zval0, D, vs0, xdts)
    Sz = sum(zval)

    # Sample theta from Normal distribution
    thetar = mvrv(mu, np.dot(sig[0], BSIG))
    thetat = np.zeros(nl)
    thetat[index] = thetar
    theta = np.vstack(thetat)
    compute_time = np.zeros(MCMC)
    for i in range(1, MCMC):    
        # print(i)
        if i % 50 == 0:
            print('MCMC, iteration-{}'.format(i))
        
        t1 = default_timer()
        # sample z from the Bernoulli distribution:
        zr = np.zeros(nl) # instantaneous latent vector (z_i):
        zr = zval
        for j in range(nl):
            ztemp0 = zr
            ztemp0[j] = 0
            if np.mean(ztemp0) == 0:
                PZ0 = pyzv0(xdts, N, asig, bsig)
            else:
                vst0 = vs[i-1]
                PZ0 = pyzv(D, ztemp0, vst0, N, xdts, asig, bsig)
            
            ztemp1 = zr
            ztemp1[j] = 1      
            vst1 = vs[i-1]
            PZ1 = pyzv(D, ztemp1, vst1, N, xdts, asig, bsig)
            
            zeta = PZ0 - PZ1  
            zeta = p0[i-1]/( p0[i-1] + np.exp(zeta)*(1-p0[i-1]))
            zr[j] = bern(1, p = zeta, size = None)
        
        zval = zr
        zstore = np.append(zstore, np.vstack(zval), axis = 1)
        
        # sample sig^2 from inverse Gamma:
        asiggamma = asig+0.5*N
        temp = np.matmul(np.matmul(mu.T, LA2.inv(BSIG + eps)), mu)
        bsiggamma = bsig+0.5*(np.dot(xdts.T, xdts) - temp)
        sig[i] = 1/IG(asiggamma, 1/bsiggamma) # inverse gamma RVs
        
        # sample vs from inverse Gamma:
        avvs = av+0.5*Sz
        bvvs = bv+(np.matmul(np.matmul(thetar.T, LA2.inv(Aor)), thetar))/(2*sig[i])
        vs[i] = 1/IG(avvs, 1/bvvs) # inverse gamma RVs
        
        # sample p0 from Beta distribution:
        app0 = ap+Sz
        bpp0 = bp+nl-Sz # Here, P=nl (no. of functions in library)
        p0[i] = beta(app0, bpp0)
        # or, np.random.beta()
        
        # Sample theta from Normal distribution:
        vstheta = vs[i]
        mu, BSIG, Aor, index = sigmu(zval, D, vstheta, xdts)
        Sz = sum(zval)
        thetar = mvrv(mu, np.dot(sig[i], BSIG))
        thetat = np.zeros(nl)
        thetat[index] = thetar
        theta = np.append(theta, np.vstack(thetat), axis = 1)
        
        t2 = default_timer()
        # print('Iteration-{}, Time-{}'.format(i, t2-t1))
        compute_time[i] = t2-t1
    
    # Marginal posterior inclusion probabilities (PIP):
    zstore_all = zstore
    zstore = zstore[:, burn_in:] # discard the first 'burn_in' samples
    Zmean = np.mean(zstore, axis=1)

    # Post processing:
    theta_all = theta
    theta = theta[:, burn_in:] # discard the first 'burn_in' samples
    mut = np.mean(theta, axis=1)
    sigt = np.cov(theta, bias = False)
    
    return zstore, Zmean, theta, mut, sigt, zstore_all, theta_all, compute_time


"""
# Bayesian Interference:
"""
class BayInt(object):

    def __init__(self, D, xdt, eps=0.0000001):
        super(BayInt, self).__init__()
        """
        Class for removing mean and standard deviation from data.
        
        D : matrix, the design matrix of library functions.
        xdt : array, the label vector.
        """
        
        self.mean_xdt = np.mean(xdt)
        self.mean_d = np.outer(np.ones(len(xdt)), np.mean(D, axis=0))
        self.std_d = np.diag(np.mean(D, axis=0) + eps)
    
    def encode(self, lib, target):
        lib = np.matmul((lib - self.mean_d)/ LA.inv(self.std_d))
        target  = (target - self.mean_xdt) 
        return lib, target
    
    def decode(self, mean, cov):
        cov = np.matmul( np.matmul( LA.inv(self.std_d), cov), cov)
        mean  = np.dot(LA.inv(self.std_d), mean)
        return mean, cov


"""
# Residual variance:
"""
def res_var(D, xdts):
    """
    Finds the residual variance of regression error.
    
    Parameters
    ----------
    D : matrix, the design matrix of library functions.
    xdts : array, the label vector.

    Returns
    -------
    scalar, the residual variance.
    """
    theta1 = np.dot(LA.pinv(D), xdts)
    error = xdts - np.matmul(D, theta1)
    err_var = np.var(error)
    
    return np.sqrt(err_var)


"""
# Initial latent vector finder:
"""
def latent_fb_1at1(nl, D, xdts):
    """
    Initializes the latent indicator variable vector.
    
    Parameters
    ----------
    nl : scalar, total number of library basses.
    D : matrix, the design matrix of library functions.
    xdts : array, the label vector.

    Returns
    -------
    zint: The initial latent indicator variable.
    """
    # Forward finder:
    zint = np.zeros(nl)
    theta = np.matmul(LA.pinv(D), xdts)
    index = np.array(np.where(zint != 0))[0]
    index = np.reshape(index,-1) # converting to 1-D array,
    Dr = D[:, index]
    thetar = theta[index]
    err = MSE(xdts, np.dot(Dr, thetar))
    for i in range(0, nl):
        index = i
        Dr = D[:, index]
        thetar = theta[index]
        err = np.append(err, MSE(xdts, np.dot(Dr, thetar)) )
        if err[i+1] <= err[i]:
            zint[index] = 1
        else:
            zint[index] = 0
    
    # Backward finder:
    index = np.array(np.where(zint != 0))
    index = np.reshape(index,-1) # converting to 1-D array,
    Dr = D[:, index]
    thetar = theta[index]
    err = MSE(xdts, np.dot(Dr, thetar))
    ind = 0
    for i in range(nl-1, -1, -1):
        index = ind
        Dr = D[:, index]
        thetar = theta[index]
        err = np.append(err, MSE(xdts, np.dot(Dr, thetar)) )
        if err[ind+1] <= err[ind]:
            zint[index] = 1
        else:
            zint[index] = 0
        ind = ind + 1
    
    # for the states
    zint[[0, 1]] = [1, 1]
    return zint


def latent_fb_marginal(nl, D, xdts):
    """
    Initializes the latent indicator variable vector.
    It only uses forward serach.
    
    Parameters
    ----------
    nl : scalar, total number of library basses.
    D : matrix, the design matrix of library functions.
    xdts : array, the label vector.

    Returns
    -------
    zint: The initial latent indicator variable.
    """
    # Forward finder:
    zint = np.zeros(nl)
    theta = np.matmul(LA.pinv(D), xdts)
    index = np.array(np.where(zint != 0))[0]
    index = np.reshape(index,-1) # converting to 1-D array,
    Dr = D[:, index]
    thetar = theta[index]
    err = MSE(xdts, np.dot(Dr, thetar))
    for i in range(0, nl):
        index = i
        Dr = D[:, index]
        thetar = theta[index]
        err = np.append(err, MSE(xdts, np.dot(Dr, thetar)) )
        if err[i+1] <= err[i]:
            zint[index] = 1
        else:
            zint[index] = 0
    return zint


def latent_fb_dual(nl, D, xdts):
    """
    Initializes the latent indicator variable vector.
    The way it is different from "latent_fb_1at1" is that it sequentially grows 
    the library and checks the significance of inclusion of new basis in the old
    library.
    
    Parameters
    ----------
    nl : scalar, total number of library basses.
    D : matrix, the design matrix of library functions.
    xdts : array, the label vector.

    Returns
    -------
    zint: The initial latent indicator variable.
    """
    # Forward grid finder:
    theta = np.matmul(LA.pinv(D), xdts)
    zint = []
    thetar = np.zeros(nl)
    err = MSE(xdts, np.dot(D, thetar))
    for i in range(0, nl):
        if i == 0:
            index = 0
        else:
            index = np.array(np.where(zint != 0))[0]
        
        Dr = D[:, index]
        if i != 0:
            Dr = np.hstack((Dr, np.vstack(D[:, i])))
        
        thetar = theta[index]
        if i != 0:
            thetar = np.append(thetar, theta[i])
        err = np.append(err, MSE(xdts, np.dot(Dr, thetar)) )
        if err[i+1] <= err[i]:
            zint = np.append(zint, 1)
        else:
            zint = np.append(zint, 0)
    
    if np.sum(zint) != 0:   # To check, whether all basis functions are non-active.
        # Backward marginal search:
        index = np.array(np.where(zint != 0))[0]
        thetar = theta[index]
        Dr = D[:, index]
        err = MSE(xdts, np.dot(Dr, thetar))
        err_0 = err     # for referencing, whenever Z_i == 0,
        ind = 0
        for i in range(nl-1, -1, -1):
            if zint[i] == 1:
                Drr = np.delete(D, i, axis=1)
                thetarr = np.delete(theta, i)
                zintr = np.delete(zint, i)
                index = np.array(np.where(zintr != 0))[0]
                Drr = Drr[:, index]
                thetarr = thetarr[index]
                
                err = np.append(err, MSE(xdts, np.dot(Drr, thetarr)) )
            else:
                err = np.append(err, err_0)
                
            if err[ind+1] <= err[ind]:
                zint[i] = 0
            else:
                zint[i] = 1
            ind = ind+1
    return zint


def nonequal_sum(mat):
    """
    It removes the identical basis functions from the parameter matrix
    
    Parameters
    ----------
    mat : matrix, the observation matrix.

    Returns
    -------
    The matrix with identical basis functions removed.
    """
    for i in range(1, mat.shape[1]):
        mat[ mat[:,i-1] !=0, i ] = 0
    return np.array(mat)

def reshape(x, shape_0, shape_1):
    """
    It reshapes a matrix to a vector
    
    Parameters
    ----------
    x : matrix, the matrix to be reshaped.
    shape_0 : integer, first dimension of shape.
    shape_1 : integer, second dimension of shape.

    Returns
    -------
    Vector, the reshaped matrix
    """
    var = []
    for i in range(shape_0):
        for j in range(shape_1):
            var.append(x[i,j,:])
    return np.array(var)

def rebuild(x, shape_0, shape_1):
    """
    It reconstruct a matrix from a vector
    
    Parameters
    ----------
    x : vector, the vector to be converted to matrix.
    shape_0 : first dimension of reconstructed matrix.
    shape_1 : second dimension of reconstructed matrix.

    Returns
    -------
    var : the retrieved matrix 
    """
    var = np.zeros((shape_0, shape_1, x.shape[-1]))
    for i in range(shape_0):
        for j in range(shape_1):
            var[i,j,:] = x[(shape_0*i) + (shape_1-(shape_1-j)), :]
    return var
    
