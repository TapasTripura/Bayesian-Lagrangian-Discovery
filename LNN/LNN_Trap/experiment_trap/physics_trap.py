# Generalized Lagrangian Networks | 2020
# Miles Cranmer, Sam Greydanus, Stephan Hoyer (...)

import jax
import jax.numpy as jnp
from jax import jit

@jit
def kinetic_energy(q, q_dot, m=1, wc=100, wz=10):
    (x1, x2, x3), (y1, y2, y3) = q, q_dot
    T = 0.5 * (y1**2 + y2**2 + y3**2)
    return T

@jit
def potential_energy(q, q_dot, m=1, wc=100, wz=10):
    (x1, x2, x3), (y1, y2, y3) = q, q_dot
    V = 0.25 * wz**2 * (x1**2 + x2**2 - 2*x3**2) + 0.5 * wc * (x1*y2 - y1*x2)
    return V 

# Structural lagrangian
@jit
def lagrangian_fn(q, q_dot, m=1, wc=100, wz=10):
    (x1, x2, x3), (y1, y2, y3) = q, q_dot
    T = kinetic_energy(q, q_dot, m=1, wc=100, wz=10)
    V = potential_energy(q, q_dot, m=1, wc=100, wz=10)
    return T - V

# Structural lagrangian
@jit
def hamiltonian_fn(q, q_dot, m=1, wc=100, wz=10):
    (x1, x2, x3), (y1, y2, y3) = q, q_dot
    T = kinetic_energy(q, q_dot, m=1, wc=100, wz=10)
    V = potential_energy(q, q_dot, m=1, wc=100, wz=10)
    return T + V
  
# Structural dynamics 
@jit
def analytical_fn(state, t=0, m=1, wc=100, wz=10):
    x1, x2, x3, y1, y2, y3 = state
    g1 = wc * y2 + 0.5 * wz**2 * x1 
    g2 = -wc * y1 + 0.5 * wz**2 * x2
    g3 = -wz**2 * x3
    return jnp.stack([y1, y2, y3, g1, g2, g3])
    
