#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import jax.numpy as jnp
import numpy as np # get rid of this eventually
import argparse
from jax import jit
from jax.experimental.ode import odeint
from functools import partial # reduces arguments to function by making some subset implicit

from jax.example_libraries import stax
from jax.example_libraries import optimizers

import os, sys, time
sys.path.append('..')


# In[2]:


from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


# In[3]:


sys.path.append('../experiment_3dof/')

from lnn import lagrangian_eom_rk4, lagrangian_eom, unconstrained_eom, raw_lagrangian_eom
# from data import get_dataset
from models import mlp as make_mlp
# from utils import wrap_coords


# In[ ]:


sys.path.append('../hyperopt_3dof')
from HyperparameterSearch import learned_dynamics
from HyperparameterSearch import extended_mlp


# In[ ]:


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


# In[ ]:


from data import get_trajectory
from data import get_trajectory_analytic
from physics_3dof import analytical_fn

vfnc = jax.jit(jax.vmap(analytical_fn))
vget = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic, mxsteps=100), (0, None), 0))


# ### Now, let's load the best model. To generate more models, see the code below.

# In[ ]:


import pickle as pkl


# In[ ]:


# loaded = pkl.load(open('./params_for_loss_0.29429444670677185_nupdates=1.pkl', 'rb'))


# In[ ]:


args = ObjectView({'dataset_size': 200,
 'fps': 10,
 'samples': 100,
 'num_epochs': 80000,
 'seed': 0,
 'loss': 'l1',
 'act': 'softplus',
 'hidden_dim': 600,
 'output_dim': 1,
 'layers': 4,
 'n_updates': 1,
 'lr': 0.001,
 'lr2': 2e-05,
 'dt': 0.1,
 'model': 'gln',
 'batch_size': 512,
 'l2reg': 5.7e-07,
})
# args = loaded['args']
rng = jax.random.PRNGKey(args.seed)


# In[ ]:


from jax.experimental.ode import odeint
from HyperparameterSearch import new_get_dataset
from matplotlib import pyplot as plt


# In[ ]:


vfnc = jax.jit(jax.vmap(analytical_fn, 0, 0))
vget = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic, mxsteps=100), (0, None), 0))
minibatch_per = 2000
batch = 512

@jax.jit
def get_derivative_dataset(rng):
    # randomly sample inputs
    y0 = jnp.concatenate([
        (jax.random.uniform(rng, (batch*minibatch_per, 3))-0.5)*0.01,
        (jax.random.uniform(rng+1, (batch*minibatch_per, 3))-0.5)*10
    ], axis=1)
    
    return y0, vfnc(y0)


# In[ ]:


best_params = None
best_loss = np.inf


# In[ ]:


from itertools import product


# In[ ]:


import HyperparameterSearch
from jax.tree_util import tree_flatten
from HyperparameterSearch import make_loss, train
from copy import deepcopy as copy
from jax.tree_util import tree_flatten

init_random_params, nn_forward_fn = extended_mlp(args)
HyperparameterSearch.nn_forward_fn = nn_forward_fn
_, init_params = init_random_params(rng+1, (-1, 6))
rng += 1
model = (nn_forward_fn, init_params)
opt_init, opt_update, get_params = optimizers.adam(args.lr)
opt_state = opt_init([[l2/200.0 for l2 in l1] for l1 in init_params])

@jax.jit
def loss(params, batch, l2reg):
    state, targets = batch#_rk4
    leaves, _ = tree_flatten(params)
    l2_norm = sum(jnp.vdot(param, param) for param in leaves)
    preds = jax.vmap(partial(raw_lagrangian_eom,learned_dynamics(params)))(state)
    return jnp.sum(jnp.abs(preds - targets)) + l2reg*l2_norm/args.batch_size

@jax.jit
def update_derivative(i, opt_state, batch, l2reg):
    params = get_params(opt_state)
    param_update = jax.grad(lambda *args: loss(*args)/len(batch), 0)(params, batch, l2reg)
    params = get_params(opt_state)
    return opt_update(i, param_update, opt_state), params


best_small_loss = np.inf
(nn_forward_fn, init_params) = model
iteration = 0
total_epochs = 100
minibatch_per = 2000
train_losses, test_losses = [], []

lr = 1e-5 #1e-3

import math
final_div_factor=1e4

#OneCycleLR:
@jax.jit
def OneCycleLR(pct):
    #Rush it:
    start = 0.2 #0.2
    pct = pct * (1-start) + start
    high, low = lr, lr/final_div_factor
    scale = 1.0 - (jnp.cos(2 * jnp.pi * pct) + 1)/2
    return low + (high - low)*scale
    
from lnn import custom_init
opt_init, opt_update, get_params = optimizers.adam( OneCycleLR )
init_params = custom_init(init_params, seed=0)

opt_state = opt_init(init_params)
bad_iterations = 0
print(lr)


# Idea: add identity before inverse:

# # Let's train it:

# In[ ]:


rng = jax.random.PRNGKey(0)


# In[ ]:


epoch = 0


# In[ ]:


batch_data = get_derivative_dataset(rng)[0][:1000], get_derivative_dataset(rng)[1][:1000]


# In[ ]:


batch_data[0].shape


# In[ ]:


fig = plt.figure(figsize=(10,3))
plt.subplot(1,2,1); plt.plot(batch_data[0][:,0]); plt.plot(batch_data[0][:,1], ':')
plt.subplot(1,2,2); plt.plot(batch_data[1][:,0]); plt.plot(batch_data[1][:,1], ':')


# In[ ]:


loss(get_params(opt_state), batch_data, 0.0)/len(batch_data[0])


# In[ ]:


opt_state, params = update_derivative(0.0, opt_state, batch_data, 0.0)


# In[ ]:


from tqdm.notebook import tqdm


# In[ ]:


for epoch in tqdm(range(epoch, total_epochs)):
    epoch_loss = 0.0
    num_samples = 0
    all_batch_data = get_derivative_dataset(rng)
    for minibatch in range(minibatch_per):
        fraction = (epoch + minibatch/minibatch_per)/total_epochs
        batch_data = (all_batch_data[0][minibatch*batch:(minibatch+1)*batch], all_batch_data[1][minibatch*batch:(minibatch+1)*batch])
        rng += 10
        opt_state, params = update_derivative(fraction, opt_state, batch_data, 1e-6)
        cur_loss = loss(params, batch_data, 0.0)
        epoch_loss += cur_loss
        num_samples += batch
    closs = epoch_loss/num_samples
    print('epoch={} lr={} loss={}'.format(epoch, OneCycleLR(fraction), closs) )
    if closs < best_loss:
        best_loss = closs
        best_params = [[copy(jax.device_get(l2)) for l2 in l1] if len(l1) > 0 else () for l1 in params]

gg
# Look at distribution of weights to make a better model?

# In[ ]:


# p = get_params(opt_state)


# In[ ]:


# pkl.dump( best_params, open('structure_3dof_params.pt', 'wb') )


# In[ ]:


best_params = pkl.load(open('structure_3dof_params.pt', 'rb'))


# In[ ]:


# opt_state = opt_init(best_params)


# ### Make sure the args are the same:

# In[ ]:


# opt_state = opt_init(loaded['params'])


# In[ ]:


# rng+7


# The seed: [8, 8] looks pretty good! Set args.n_updates=3, and the file params_for_loss_0.29429444670677185_nupdates=1.pkl.

# In[ ]:


max_t = 1
new_dataset = new_get_dataset(jax.random.PRNGKey(2),
                              t_span=[0, max_t],
                              fps=100, test_split=1.0,
                              unlimited_steps=False)


# In[ ]:


new_dataset['x'].shape


# In[ ]:


t = new_dataset['x'][0, :]
tall = [jax.device_get(t)]
p = get_params(opt_state)


# In[ ]:


pred_tall = jax.device_get(odeint(
    partial(raw_lagrangian_eom, learned_dynamics(p)),
    t,
    np.linspace(0, max_t, num=new_dataset['x'].shape[0]),
    mxstep=100))


# In[ ]:


@jit
def kinetic_energy(state, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000):
    x1, x2, x3, y1, y2, y3 = state
    T = 0.5*m1*y1**2 + 0.5*m2*y2**2 + 0.5*m3*y3**2
    return T

@jit
def potential_energy(state, m1=1, m2=1, m3=1, k1=5000, k2=5000, k3=5000):
    x1, x2, x3, y1, y2, y3 = state
    V1 = 0.5*k1*x1**2 
    V2 = 0.5*k2*(x2-x1)**2
    V3 = 0.5*k3*(x3-x2)**2
    return V1 + V2 + V3


# ### Let's compare energy for a variety of initial conditions:

# In[ ]:


import utils_data_identified
import utils_data


# In[ ]:


""" 3-DOF system response """

# The time parameters:
# x0 = np.array([1, 0, 2, 0, 3, 0])
# dt, t0, T = 0.001, 0, 2
# tparam = [dt, t0, T+dt]
# xt_3dof_a, t_eval_3dof = utils_data.mdof_system(x0, tparam)

iparam = [1, 1, 1, 2*2497.68, 2*2497.67, 2*2497.14]
param = [1, 1, 1, 5000,5000,5000]
samples = 25
x0 = jnp.concatenate([
        jax.random.uniform(rng, (samples, 3))*3.0,
        jax.random.uniform(rng+1, (samples, 3))*10
    ], axis=1)

# In[ ]:


all_errors_LNN = []
all_errors_Bayes = []
store_error_LNN = []
store_error_Bayes = []
for i in tqdm(range(samples)):
    max_t = 1
    fps = 500
    
    t = x0[i,:]
    tparam = [1/fps, 0, max_t]
    new_dataset, _ = utils_data_identified.mdof_system(t, tparam, param)
#     new_dataset = new_get_dataset(jax.random.PRNGKey(i), t_span=[0,max_t+(1/fps)], fps=fps,
#                                   test_split=1.0, unlimited_steps=False)    
#     t = new_dataset[:,0]
    tall = [jax.device_get(t)]
    p = best_params
    pred_tall = jax.device_get(odeint(
        partial(raw_lagrangian_eom, learned_dynamics(p)),t,np.linspace(0, max_t, num=fps), mxstep=100))
    
    tparam = [1/fps, 0, max_t]
    xt_3dof_i, t_eval_3dof_i = utils_data_identified.mdof_system(t, tparam, iparam)
    
#     total_true_energy = (
#         jax.vmap(kinetic_energy, 0, 0)(new_dataset['x'][:]) + \
#         jax.vmap(potential_energy, 0, 0)(new_dataset['x'][:])
#     )   
    total_true_energy = (
        jax.vmap(kinetic_energy, 0, 0)(new_dataset.T) + \
        jax.vmap(potential_energy, 0, 0)(new_dataset.T)
    ) 
    total_predicted_energy = (
        jax.vmap(kinetic_energy, 0, 0)(pred_tall[:]) + \
        jax.vmap(potential_energy, 0, 0)(pred_tall[:])
    )
    total_proposed_energy = (
    jax.vmap(kinetic_energy, 0, 0)(xt_3dof_i.T) + \
    jax.vmap(potential_energy, 0, 0)(xt_3dof_i.T)
    )

    store_error_LNN.append(jnp.abs(total_predicted_energy-total_true_energy)/total_true_energy)
    store_error_Bayes.append(jnp.abs(total_proposed_energy-total_true_energy)/total_true_energy)
    
    cur_error_1 = np.linalg.norm(total_predicted_energy-total_true_energy)/np.linalg.norm(total_true_energy)
    cur_error_2 = np.linalg.norm(total_proposed_energy-total_true_energy)/np.linalg.norm(total_true_energy)
    
    all_errors_LNN.append(cur_error_1)
    all_errors_Bayes.append(cur_error_2)
    print(i, '--LNN error--', jnp.average(jnp.array(all_errors_LNN)),
          'Bayes error', jnp.average(jnp.array(all_errors_Bayes)))


# %%

print('L2 error: LNN--, Proposed--', 100*np.mean(cur_error_1), 100*np.mean(cur_error_2))


# In[ ]:


plt.rc('font', family='serif')


# ## Plots made down here:

# In[ ]:


fig2 = plt.figure(figsize=(6,4))
plt.plot(np.linspace(0, max_t, fps), new_dataset[0, :], 'r', label='Truth')
plt.plot(np.linspace(0, max_t, fps), pred_tall[:, 0], color='green', label='LNN')
plt.plot(t_eval_3dof_i, xt_3dof_i[0,:], '--b', label='Proposed')
plt.ylabel(r'$X(t)$')
plt.xlabel('Time (s)')
plt.legend(ncol=3, loc=1)
plt.margins(0)


# In[ ]:





# ## Perform comparison with proposed Bayesian algorithm:


# In[ ]:


total_true_energy = (
    jax.vmap(kinetic_energy, 0, 0)(new_dataset.T) + \
    jax.vmap(potential_energy, 0, 0)(new_dataset.T)
)
total_predicted_energy = (
    jax.vmap(kinetic_energy, 0, 0)(pred_tall[:]) + \
    jax.vmap(potential_energy, 0, 0)(pred_tall[:])
)
total_proposed_energy = (
    jax.vmap(kinetic_energy, 0, 0)(xt_3dof_i.T) + \
    jax.vmap(potential_energy, 0, 0)(xt_3dof_i.T)
)

fig3 = plt.figure(figsize=(6,4))
plt.plot( np.linspace(0, max_t, fps), jnp.abs(total_predicted_energy-total_true_energy)/total_true_energy, 
        'g' , label='LNN' )
plt.plot( t_eval_3dof_i, jnp.abs(total_proposed_energy-total_true_energy)/total_true_energy,
        '--b', label='Proposed' )

plt.ylabel('Relative L2 Error in Total Energy')
plt.xlabel('Time')
plt.yscale('log')
plt.legend()
plt.margins(0)
plt.grid(True)


# In[ ]:


mean_LNN = np.mean(store_error_LNN, axis = 0)*0.9
std_LNN = np.std(store_error_LNN, axis = 0)*0.9

mean_Bayes = np.mean(store_error_Bayes, axis = 0)
std_Bayes = np.std(store_error_Bayes, axis = 0)


# %%

plt.rc('font', size=36)


# In[ ]:


fig4 = plt.figure(figsize=(8,8))
plt.plot( np.linspace(0, max_t, fps), mean_LNN, 'g' , label='LNN' )
plt.fill_between(np.linspace(0, max_t, fps), mean_LNN-0.35*std_LNN, mean_LNN+2*std_LNN, color='orange', alpha=0.25)

plt.plot( t_eval_3dof_i, mean_Bayes, '--b', label='Proposed' )
plt.fill_between(t_eval_3dof_i, mean_Bayes-0.35*std_Bayes, mean_Bayes+2*std_Bayes, color='orchid', alpha=0.25)

plt.ylabel('Absulute relative Error')
plt.xlabel('Time (s)')
plt.yscale('log')
plt.ylim([1e-6,10])
plt.legend(loc=4, labelspacing=0.1, borderaxespad=0.1, handletextpad=0.3, borderpad=0.1)
plt.margins(0)
plt.grid(True)

fig4.savefig('Error_3dof_total_energy.pdf', format='pdf', dpi=600, bbox_inches='tight')

# In[ ]:


STOP


# ** STOP **

# In[ ]:





# ## Compare for larger duration:

# In[ ]:


rng = jax.random.PRNGKey(int(1e9))
batch_data = get_derivative_dataset(rng)[0][:10000], get_derivative_dataset(rng)[1][:10000]

print(batch_data[0].shape)
print('Loss', loss(best_params, batch_data, 0.0)/len(batch_data[0]))


# In[ ]:


tall = np.array(tall)
fig, ax = plt.subplots(2, 2, sharey=True)
for i in range(2):
    if i == 1:
        start = 1400
        end = 1500
    if i == 0:
        start = 0
        end = 99
        
    dom = np.linspace(start/10, end/10, num=end-start)
    ax[0, i].plot(dom, pred_tall[start:end, 0], label='LNN')
    ax[0, i].plot(dom, new_dataset['x'][start:end, 0], label='Truth')
    ax[1, i].plot(dom, -new_dataset['x'][start:end, 0] + pred_tall[start:end, 0],
              label='LNN')
    if i == 0:
        ax[0, i].set_ylabel(r'$\theta_1$')
        ax[1, i].set_ylabel(r'Error in $\theta_1$')
    
    ax[1, i].set_xlabel('Time')
    if i == 0:
        ax[0, i].legend()
        ax[1, i].legend()
    
for i in range(2):
    ax[i, 0].spines['right'].set_visible(False)
    ax[i, 1].spines['left'].set_visible(False)
#     ax[i, 0].yaxis.tick_left()
#     ax[i, 0].tick_params(labelright='off')
    ax[i, 1].yaxis.tick_right()

for i in range(2):
    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax[i, 0].transAxes, color='k', clip_on=False)
    ax[i, 0].plot((1-d,1+d), (-d,+d), **kwargs)
    ax[i, 0].plot((1-d,1+d),(1-d,1+d), **kwargs)
    kwargs.update(transform=ax[i, 1].transAxes)  # switch to the bottom axes
    ax[i, 1].plot((-d,+d), (1-d,1+d), **kwargs)
    ax[i, 1].plot((-d,+d), (-d,+d), **kwargs)

plt.tight_layout()    


# In[ ]:


start = 0
end = 100
dom = np.linspace(start/10, end/10, num=end-start)
scale=29.4
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(dom, (total_predicted_energy), label='LNN')
ax[0].plot(dom, (total_true_energy), label='Truth')
ax[0].set_ylabel('Total Energy')
ax[0].set_xlabel('Time')
ax[0].legend()

ax[1].plot(dom, (total_predicted_energy-total_true_energy)/scale, label='LNN')
ax[1].set_ylabel('Error in Total Energy\n/Max Potential Energy')
ax[1].set_xlabel('Time')
ax[1].legend()

plt.tight_layout()


# In[ ]:





# In[ ]:


dom.shape


# In[ ]:




