# Generalized Lagrangian Networks | 2020
# Miles Cranmer, Sam Greydanus, Stephan Hoyer (...)

import jax
import jax.numpy as jnp
from jax import jit

@jit
def kinetic_energy(q, q_dot, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000):
    (x1, x2, x3), (y1, y2, y3) = q, q_dot
    T = 0.5*m1*y1**2 + 0.5*m2*y2**2 + 0.5*m3*y3**2
    return T

@jit
def potential_energy(q, q_dot, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000):
    (x1, x2, x3), (y1, y2, y3) = q, q_dot
    V1 = 0.5*k1*x1**2 
    V2 = 0.5*k2*(x2-x1)**2
    V3 = 0.5*k3*(x3-x2)**2
    return V1 + V2 + V3

# Structural lagrangian
@jit
def lagrangian_fn(q, q_dot, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000):
    (x1, x2, x3), (y1, y2, y3) = q, q_dot
    T = kinetic_energy(q, q_dot, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000)
    V = potential_energy(q, q_dot, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000)
    return T - V

# Structural lagrangian
@jit
def hamiltonian_fn(q, q_dot, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000):
    (x1, x2, x3), (y1, y2, y3) = q, q_dot
    T = kinetic_energy(q, q_dot, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000)
    V = potential_energy(q, q_dot, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000)
    return T + V
  
# Structural dynamics 
@jit
def analytical_fn(state, t=0, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000):
    x1, x2, x3, y1, y2, y3 = state
    g1 = -k1*x1 -k2*(x1-x2)
    g2 = -k2*(x2-x1) -k3*(x2-x3)
    g3 = -k3*(x3-x2)
    return jnp.stack([y1, y2, y3, g1, g2, g3])
    
