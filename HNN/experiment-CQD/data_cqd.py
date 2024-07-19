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
    a, b, c = 1000, 5000, 90000
    q, p = np.split(coords,2)
    H = 0.5*p**2 + 0.5*a*q**2 + 0.25*b*q**4 + (1/6)*c*q**6 # CQD hamiltonian (nonlinear oscillator)
    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dpdt, dqdt = np.split(dcoords,2)
    S = np.concatenate([dqdt, -dpdt], axis=-1)
    return S

def get_trajectory(t_span=[0,1], timescale=1000, radius=None, y0=None, noise_std=0.0, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.uniform(0.1, 0.5, size=2)
    if radius is None:
        radius = np.random.rand()*0.5 + 0.1 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    cqd_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10)
    q, p = cqd_ivp['y'][0], cqd_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in cqd_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    return q, p, dqdt, dpdt, t_eval

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
            x, y, dx, dy, t = get_trajectory(**kwargs)
            xs.append( np.stack( [x, y]).T )
            dxs.append( np.stack( [dx, dy]).T )
            
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

