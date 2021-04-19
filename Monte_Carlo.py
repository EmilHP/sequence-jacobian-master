# MC module for easy distribution calculations. Used after s.s. is found with histogram method. 

import numpy as np
np.random.seed(2021) # fix seed
from numba import njit, prange
import time
from iteround import saferound

 # consav
from consav import linear_interp # for linear interpolation

# local modules

@njit
def choice(p,r): 
    ''' Find next state in Markov-Chain given cumulative transition matrix and random number'''

    i = 0
    while r > p[i]:
        i = i + 1
        
    return i

def steady_state_MC(simT,simN,b_policy,d_policy,c_policy,e_grid,b_grid,d_grid,e_ergodic,Pi_e):
    """ simulate forward to steady-state """

    np.random.seed(100)

    # unpack
    e_cum = np.cumsum(Pi_e[:],axis = 1)
    ergodic = np.asarray(saferound(e_ergodic*simN, places=0))
    Ne = len(e_grid)
    
    sim_shape = (simT,simN)
    sim_p = np.zeros(sim_shape)
    sim_b = np.zeros(sim_shape)
    sim_b[0,:] = b_policy[0,0,0] # initial state
    sim_d = np.zeros(sim_shape) 
    sim_d[0,:] = d_policy[0,0,0] # initial state
    sim_c = np.zeros(sim_shape) 
    sim_c[0,:] = c_policy[0,0,0] # initial state
    unif = np.random.uniform(size=(simT,simN)) # random uniform shocks
    ergodic = e_ergodic

    sim_b,sim_d, sim_c= run(simT,simN,b_policy,d_policy,c_policy,e_grid,b_grid,d_grid,ergodic,e_cum,Ne,sim_p,sim_b,sim_d,sim_c,unif)

    return sim_b, sim_d, sim_c

@njit(parallel=True)
def run(simT,simN,b_policy,d_policy,c_policy,e_grid,b_grid,d_grid,ergodic,e_cum,Ne,sim_p,sim_b,sim_d,sim_c,unif):
    
    # i. initial guess on productivity distribution

    for i in range(Ne):
        if i == 0:
            sim_p[0,0:int(ergodic[i])] = i
        else:       
            sim_p[0,int(np.sum(ergodic[:i])):int(np.sum(ergodic[:i]) + ergodic[i])] = i

    # ii. simulate   
    for t in range(1,simT):
        for n in prange(simN): # parallelize inner loop as time needs to be consistent

            # i. Markov Chain shock
            sim_p[t,n] = choice(e_cum[int(sim_p[t-1,n])],unif[t,n])
        
            # ii. last period d,b
            b_minus = sim_b[t-1,n]
            d_minus = sim_d[t-1,n]
        
            # iii. find current d,b by interpolation of policy functions
            sim_d[t,n] = linear_interp.interp_2d(b_grid,d_grid,d_policy[int(sim_p[t,n]),:,:],b_minus,d_minus)
            sim_b[t,n] = linear_interp.interp_2d(b_grid,d_grid,b_policy[int(sim_p[t,n]),:,:],b_minus,d_minus)
            sim_c[t,n] = linear_interp.interp_2d(b_grid,d_grid,c_policy[int(sim_p[t,n]),:,:],b_minus,d_minus)

    return sim_b, sim_d, sim_c
