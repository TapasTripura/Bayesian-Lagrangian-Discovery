# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import to_pickle, from_pickle

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

def get_trajectory(t_span=[0,1], timescale=1000, radius=None, y0=None, noise_std=0.0, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.uniform(1e-1, 1e-3, size=6)
    if radius is None:
        radius = np.random.rand()*1e-3 + 1e-2 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    trap_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
    q1, q2, q3, p1, p2, p3 = trap_ivp['y'][0], trap_ivp['y'][1], trap_ivp['y'][2], \
                                trap_ivp['y'][3], trap_ivp['y'][4], trap_ivp['y'][5]
    dydt = [dynamics_fn(None, y) for y in trap_ivp['y'].T]
    dydt = np.stack(dydt).T
    dq1dt, dq2dt, dq3dt, dp1dt, dp2dt, dp3dt = np.split(dydt,6)

    # add noise
    q1 += np.random.randn(*q1.shape)*noise_std
    q2 += np.random.randn(*q2.shape)*noise_std
    q3 += np.random.randn(*q3.shape)*noise_std
    p1 += np.random.randn(*p1.shape)*noise_std
    p2 += np.random.randn(*p2.shape)*noise_std
    p3 += np.random.randn(*p3.shape)*noise_std
    
    return q1, q2, q3, p1, p2, p3, dq1dt, dq2dt, dq3dt, dp1dt, dp2dt, dp3dt, t_eval

def get_dataset(experiment_name, save_dir, seed=0, samples=2, test_split=0.5, **kwargs):
    
    path = '{}/{}-dataset.pkl'.format(save_dir, experiment_name)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))        
        data = {'x': None, 'dx': None}
    
        # randomly sample inputs
        np.random.seed(seed)
        xs, dxs = [], []
        for s in range(samples):
            x1, x2, x3, y1, y2, y3, dx1, dx2, dx3, dy1, dy2, dy3, t = get_trajectory(**kwargs)
            xs.append( np.stack( [x1, x2, x3, y1, y2, y3]).T )
            dxs.append( np.stack( [dx1, dx2, dx3, dy1, dy2, dy3]).T )
            
        data['x'] = np.concatenate(xs)
        data['dx'] = np.concatenate(dxs).squeeze()
    
        # make a train/test split
        split_ix = int(len(data['x']) * test_split)
        split_data = {}
        for k in ['x', 'dx']:
            split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
            
        data = split_data
        to_pickle(data, path)
    return data

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field
