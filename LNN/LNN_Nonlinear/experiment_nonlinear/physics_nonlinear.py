# Generalized Lagrangian Networks | 2020
# Miles Cranmer, Sam Greydanus, Stephan Hoyer (...)

import jax
import jax.numpy as jnp
from jax import jit

@jit
def kinetic_energy(q, q_dot, m=1, a=1000, b=5000, c=90000):
    T = 0.5 * m * q_dot**2
    return T

@jit
def potential_energy(q, q_dot, m=1, a=1000, b=5000, c=90000):
    V = 0.5*a*q**2 + 0.25*b*q**4 + (1/6)*c*q**6
    return V

# Nonlinear lagrangian
@jit
def lagrangian_fn(q, q_dot, m=1, a=1000, b=5000, c=90000):
    T = kinetic_energy(q, q_dot, m=1, a=1000, b=5000, c=90000)
    V = potential_energy(q, q_dot, m=1, a=1000, b=5000, c=90000)
    return T - V

# Nonlinear lagrangian
@jit
def hamiltonian_fn(q, q_dot, m=1, a=1000, b=5000, c=90000):
    T = kinetic_energy(q, q_dot, m=1, a=1000, b=5000, c=90000)
    V = potential_energy(q, q_dot, m=1, a=1000, b=5000, c=90000)
    return T + V
  
# Nonlinear dynamics 
@jit
def analytical_fn(state, t=0, m=1, a=1000, b=5000, c=90000):
    q, q_dot = state
    g = - a*q - b*q**3 - c*q**5
    return jnp.stack([q_dot, g])
    
