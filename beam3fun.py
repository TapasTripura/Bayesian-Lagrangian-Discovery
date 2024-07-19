#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Discovering interpretable Lagrangian of dynamical systems from data
-- Authored by: Tapas Tripura and Souvik Chakraborty

-- This code contains the useful functions for Beam vibration simulation
"""

import numpy as np

def Beam3(rho,A,E,I,Le,n,bc,k_t1=0,k_t2=0,k_r1=0,k_r2=0):
    # ------------------------------------------------------------------------!
    # Euler-Bernoulli Beam 
    # spatial discretization by finite element method
    # n is the NUMBER of NODES including the left end
    # Le is the LENGTH of each ELEMENT
    # -------------------------------------------------------------------------

    # element stiffness matrix
    Me = rho*A*Le/420*np.array([[156,    22*Le,   54,     -13*Le],
                                [22*Le,  4*Le**2,  13*Le,  -3*Le**2],
                                [54,     13*Le,   156,    -22*Le],
                                [-13*Le, -3*Le**2, -22*Le, 4*Le**2]])
                       
    Ke = E*I/(Le**3)*np.array([[12,    6*Le,    -12,    6*Le],
                            [6*Le,  4*Le**2,  -6*Le,  2*Le**2],
                            [-12,   -6*Le,   12,     -6*Le],
                            [6*Le,  2*Le**2,  -6*Le,  4*Le**2]])
    
    # -------------------------------------------------------------------------
    # global stiffness matrix
    Ma = np.zeros([2*n,2*n])
    Ka = np.zeros([2*n,2*n])
    for i in range(0, 2*n-3, 2):
        Ma[i:i+4,i:i+4] = Ma[i:i+4,i:i+4] + Me
        Ka[i:i+4,i:i+4] = Ka[i:i+4,i:i+4] + Ke

    # -------------------------------------------------------------------------
    # boundary conditions !
    # bcs = 'general';
    # bcs = 'simply-supported';
    if bc == 'cantilever':
        # the left end is clamped !
        Ma = np.delete(Ma, [0,1], 1) # column delete
        Ma = np.delete(Ma, [0,1], 0) # row delete
        Ka = np.delete(Ka, [0,1], 1)
        Ka = np.delete(Ka, [0,1], 0)
        
    elif bc == 'simply-supported':
        # simply supported at two ends
        Ma = np.delete(Ma, [0,-2], 1) # first and second last column
        Ma = np.delete(Ma, [0,-2], 0) # first and second last row
        Ka = np.delete(Ka, [0,-2], 1)
        Ka = np.delete(Ka, [0,-2], 0)
        
    elif bc == 'general':
          # linear translational and rotational springs at both ends
          # E I y''' = - k_t2 * y   --- right end
          # E I y''' =   k_t1 * y   --- left end
          # E I y''  =   k_r2 * y'  --- right end
          # E I y''  = - k_r1 * y'  --- left end 
        Ka[0,0] = Ka(1,1) + k_t1;
        Ka[1,1] = Ka(2,2) + k_r1;
        Ka[-2,-2] = Ka(-2,-2) + k_t2; 
        Ka[-1,-1] = Ka[-1,-1] + k_r2;

    # -------------------------------------------------------------------------
    # natural frequency
    w2, _ = np.linalg.eig( np.matmul( np.linalg.inv(Ma),Ka ) )
    w2 = np.sort(w2)
    omega = np.sqrt(w2)
    lambdaL = np.sqrt(omega)*(rho*A/E/I)**(1/4)
    return Ma, Ka, omega, lambdaL

def Newmark(M,C,K,F,D0,V0,dt,T,Beta=1/4,Gamma=1/2):
    # Newmark method for linear time invariant system
    # **************    M*Y''(t)+C*Y'(t)+K*Y(t)=F(t)    **********************
    # D0  - initial displacement
    # V0  - initial velocity
    # dt  - time increment
    # T   - final time 
    # -------------------------------------------------------------------------
    # Reference
    # Dynamics of Structures. Chopra A K
    # Dynamics of Structures. Clough R W, Penzien J
    # -------------------------------------------------------------------------

    # integration constant
    c1 = 1/Beta/dt**2
    c2 = Gamma/Beta/dt
    c3 = 1/Beta/dt
    c4 = 1/2/Beta-1
    c5 = Gamma/Beta-1
    c6 = (Gamma/2/Beta-1)*dt
    c7 = (1-Gamma)*dt
    c8 = Gamma*dt
    # -------------------------------------------------------------------------
    A0 = np.dot( np.linalg.inv(M), (F(0)- np.dot(K,D0)- np.dot(C,V0)) )      # initial acceleration
    n = int(T/dt+1)
    m = len(D0)
    D = np.zeros([m,n])
    V = np.zeros([m,n])
    A = np.zeros([m,n])
    D[:, 0] = D0
    V[:, 0] = V0
    A[:, 0] = A0
    # -------------------------------------------------------------------------
    Kbar = c1*M + c2*C + K          # linear time-invariant system
    for i in range(n-1):
        Da = D[:, i]
        Va = V[:, i]
        Aa = A[:, i]
        Fbar = F(i*dt) + np.dot(M, (c1*Da+c3*Va+c4*Aa)) + np.dot(C, (c2*Da+c5*Va+c6*Aa))
        D[:,i+1] = np.matmul(np.linalg.inv(Kbar), Fbar)
        A[:,i+1] = c1*(D[:,i+1]-Da) -c3*Va -c4*Aa
        V[:,i+1] = Va +c7*Aa +c8*A[:,i+1]
    return D, V, A
