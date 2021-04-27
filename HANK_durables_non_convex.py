import numpy as np
from numba import vectorize, njit, prange

import utils
from het_block import het
from simple_block import simple

import time
import numpy as np
from numba import njit, prange


# consav package
from consav import ModelClass, jit # baseline model class and jit
from consav import linear_interp # for linear interpolation
from consav import golden_section_search # for optimization in 1D
from consav.grids import nonlinspace # grids


'''Part 1: HA block'''

@het(exogenous='Pi_e', policy=['nn','b'], backward=['V_keep', 'V_adj'])  # order as in grid!
def household(V_keep_p,V_adj_p,Pi_e_p,beta,r,sigma,tau,delta,sigma_eps,
                Np,Nn,Nm,Nx,Nb,e_ergodic,e_grid,nn_grid,m_grid,x_grid,b_grid,n_max,kappa,Div,P,P_d,P_n,W,N,tau_e,Tax):

    """Single backward iteration step using NVFI for households with separable CRRA utility."""
    d_ubar = 1e-2
    # a. grid extensions
    r_grid = b_grid.copy()
    r_grid[:] = r
    r_grid[b_grid<0] = r + kappa
    R = 1+r
    P_n_p = P_n/P
    P_d_p = P_d/P
    tau_a = 0
    lump = (1 + tau_a*(b_grid[np.newaxis,:]/1-1) + tau_e*(e_grid[:,np.newaxis] - 1))*Div

    grid_z = (W/P) * e_grid * N + lump[:,0]

    if Tax is None:
        tax = 0
    else:
        tax = Tax / np.sum(e_ergodic * e_grid) * e_grid
        grid_z -= tax

    # c. compute time step 
    dg, c, b, nn, V_keep, V_adj = time_iteration(V_keep_p,V_adj_p,Pi_e_p,beta,
    R,sigma,d_ubar,tau,delta,sigma_eps,Np,Nn,Nm,Nx,Nb,grid_z,nn_grid,m_grid,x_grid,b_grid,n_max,P_n_p,P_d_p)
    c_d = dg - nn_grid[np.newaxis,:,np.newaxis]
    #n = dg # test

    return V_keep, V_adj, nn, b, dg, c_d, c


@njit(fastmath=True)
def utility(c,d,d_ubar,sigma):
    dtot = d+d_ubar
    result = np.log(c) + dtot**(1-sigma)/(1-sigma)

    return result

@njit
def obj_last_period(d,x,sigma,d_ubar,P_n_p,P_d_p): # P_n_p, P_d_p
    """ objective function in last period """
    
    # implied consumption (rest)
    c = (x-P_d_p*d)/P_n_p
    
    # value of choice
    val_of_choice = utility(c,d,d_ubar,sigma)

    return -val_of_choice

@njit(parallel=True)
def last_period(sigma,d_ubar,Np,Nn,Nm,Nx,grid_n,grid_m,grid_x,n_max,P_n_p,P_d_p): # P_n_p, P_d_p
    """ solve the problem in the last period """

    # unpack
    keep_shape = (Np,Nn,Nm)
    c_keep = np.zeros(keep_shape)
    inv_v_keep = np.zeros(keep_shape)
    
    adj_shape = (Np,Nx)
    d_adj = np.zeros(adj_shape)
    c_adj = np.zeros(adj_shape)
    inv_v_adj = np.zeros(adj_shape)
    

    # a. keep
    for i_p in prange(Np):
        for i_n in range(Nn):
            for i_m in range(Nm):
                            
                # i. states
                n = grid_n[i_n]
                m = grid_m[i_m]

                if i_m == 0: # forced c = 0 
                    c_keep[i_p,i_n,i_m] = 0
                    inv_v_keep[i_p,i_n,i_m] = 0
                    continue
                
                # ii. optimal choice
                c_keep[i_p,i_n,i_m] = m/P_n_p

                # iii. optimal value
                v_keep = utility(c_keep[i_p,i_n,i_m],n,d_ubar,sigma)
                inv_v_keep[i_p,i_n,i_m] = -1.0/v_keep

    # b. adj
    for i_p in prange(Np):
        for i_x in range(Nx):
            
            # i. states
            x = grid_x[i_x]

            if i_x == 0: # forced c = d = 0
                d_adj[i_p,i_x] = 0
                c_adj[i_p,i_x] = 0
                inv_v_adj[i_p,i_x] = 0
                continue

            # ii. optimal choices
            d_low = np.fmin(x/2,1e-8)
            d_high = np.fmin(x/P_d_p,n_max)            
            d_adj[i_p,i_x] = golden_section_search.optimizer(obj_last_period,d_low,d_high,args=(x,sigma,d_ubar,P_n_p,P_d_p),tol=1e-8)
            c_adj[i_p,i_x] = (x-P_d_p*d_adj[i_p,i_x])/P_n_p # x - P_n_p*c - P_n_d*d = 0 =>  c=(x- P_n_d*d)/P_n_p

            # iii. optimal value
            v_adj = -obj_last_period(d_adj[i_p,i_x],x,sigma,d_ubar,P_n_p,P_d_p)
            inv_v_adj[i_p,i_x] = -1.0/v_adj
        
    return c_keep, c_adj, d_adj, inv_v_keep, inv_v_adj


@njit(parallel=True)
def compute_w(beta,R,delta,inv_v_keep_p,inv_v_adj_p,Np,Nn,Nm,Nb,grid_p,grid_n,grid_m,grid_x,grid_b,p_trans,n_max,tau,sigma_eps,P_d_p):# P_d_p
    """ compute the post-decision function w """

    # unpack
    post_shape = (Np,Nn,Nb)
    inv_w = np.nan*np.zeros(post_shape)

    # loop over outermost post-decision state
    for i_p in prange(Np):

        # allocate temporary containers
        m_plus = np.zeros(Nb) # container, same lenght as grid_b
        x_plus = np.zeros(Nb)
        w = np.zeros(Nb) 
        inv_v_keep_plus = np.zeros(Nb)
        inv_v_adj_plus = np.zeros(Nb)
        
        # loop over other outer post-decision states
        for i_n in range(Nn):

            # a. permanent income and durable stock
            n = grid_n[i_n]

            # b. initialize at zero
            for i_b in range(Nb):
                w[i_b] = 0.0

            # c. loop over shocks and then end-of-period assets
            for ishock in range(Np):
                
                # i. shocks
                weight = p_trans[i_p,ishock]

                # ii. next-period income and durables
                n_plus = (1-delta)*n
                n_plus = np.fmin(n_plus,n_max) # upper bound
                p_plus = grid_p[ishock]

                # iii. prepare interpolators
                prep_keep = linear_interp.interp_3d_prep(grid_p,grid_n,p_plus,n_plus,Nb)
                prep_adj = linear_interp.interp_2d_prep(grid_p,p_plus,Nb)

                # v. next-period cash-on-hand and total resources
                for i_b in range(Nb):
        
                    m_plus[i_b] = R*grid_b[i_b]+ p_plus - grid_b[0] # borrowing allowed
                    x_plus[i_b] = m_plus[i_b] + P_d_p*(1-tau)*n_plus # *pris
                
                # vi. interpolate
                linear_interp.interp_3d_only_last_vec_mon(prep_keep,grid_p,grid_n,grid_m,inv_v_keep_p,p_plus,n_plus,m_plus,inv_v_keep_plus) # t+1 v
                linear_interp.interp_2d_only_last_vec_mon(prep_adj,grid_p,grid_x,inv_v_adj_p,p_plus,x_plus,inv_v_adj_plus) # t+1 v

                # vii. max and accumulate
                for i_b in range(Nb):
                    if inv_v_keep_plus[i_b] == 0:
                        v_keep =-9999.0
                    else:
                        v_keep=-1.0/inv_v_keep_plus[i_b]
                    if inv_v_adj_plus[i_b] == 0:
                        v_adj=-9999.0
                    else:
                        v_adj=-1.0/inv_v_adj_plus[i_b]   

                    v_max = np.fmax(v_keep,v_adj) 
                    val = v_max + sigma_eps*np.log(np.exp((v_keep-v_max)/sigma_eps) + np.exp((v_adj-v_max)/sigma_eps)) 
                    w[i_b] += weight*beta*(val)               


            # d. transform post decision value function
            for i_b in range(Nb):
                inv_w[i_p,i_n,i_b] = -1/w[i_b]
    return inv_w


#@njit(parallel=True)
def collect_policy(Np,Nn,Nm,Nb,grid_p,grid_n,grid_m,grid_x,grid_b,inv_v_keep,inv_v_adj,c_keep,c_adj,d_adj,sigma_eps,tau,delta,n_max,R,P_d_p,P_n_p): # P_d_p, P_n_p
    """solve bellman equation for keepers using nvfi"""

    # unpack output
    collected_shape = (Np,Nn,Nb)
    c_combined = np.zeros(collected_shape)
    d_combined = np.zeros(collected_shape)
    n_plus = np.zeros(collected_shape)
    a_plus = np.zeros(collected_shape)
    #inv_v = np.zeros(collected_shape)

    # loop over outer states
    for i_p in prange(Np):
        # permanent income shock
        p = grid_p[i_p]
        for i_n in range(Nn):
            
            # outer states
            n = grid_n[i_n]

            # loop over m state
            for i_b in range(Nb): # beginning of period assets
                # i. cash on hand
                m = R*grid_b[i_b] + p - grid_b[0] # cash on hand non-adjuster
                x = m + P_d_p*(1-tau)*n # adjuster cash on hand

                # ii. value functions                
                inv_v_k = linear_interp.interp_1d(grid_m,inv_v_keep[i_p,i_n,:],m)
                inv_v_a = linear_interp.interp_1d(grid_x,inv_v_adj[i_p,:],x)

                if inv_v_k == 0:
                    v_keep =-9999.0
                else:
                    v_keep = -1.0/inv_v_k
                if inv_v_a == 0:
                    v_adj=-9999.0
                else:
                    v_adj = -1.0/inv_v_a

                v_max = np.fmax(v_keep,v_adj)

                # iii. choice probability adjusting
                numerator = np.exp((v_adj-v_max)/sigma_eps)
                denominator = np.exp((v_adj-v_max)/sigma_eps) + np.exp((v_keep-v_max)/sigma_eps)
                P_adj = numerator/denominator

                # iv. combined policy functions
                # o. adjuster
                d_adj_temp = linear_interp.interp_1d(grid_x,d_adj[i_p,:],x) 
                c_adj_temp = linear_interp.interp_1d(grid_x,c_adj[i_p,:],x)

                tot_adj = P_d_p*d_adj_temp + P_n_p*c_adj_temp
                if tot_adj > x: 
                    d_adj_temp *= x/tot_adj
                    c_adj_temp *= x/tot_adj
                    a_adj_temp = 0.0
                    n_plus_adj_temp = (1-delta)*d_adj_temp
                    n_plus_adj_temp = np.fmin(n_plus_adj_temp,n_max)
                else:
                    a_adj_temp = x - tot_adj
                    n_plus_adj_temp = (1-delta)*d_adj_temp
                    n_plus_adj_temp = np.fmin(n_plus_adj_temp,n_max)

                # oo. keeper
                d_keep_temp = n
                c_keep_temp = linear_interp.interp_1d(grid_m,c_keep[i_p,i_n,:],m) 
                
                if c_keep_temp > m: 
                    c_keep_temp = m/P_n_p # P_n_p*c, have m. Use all m, P_n_p*c=m, c=m/P_n_p
                    a_keep_temp = 0.0
                    n_plus_keep_temp = (1-delta)*n
                    n_plus_keep_temp= np.fmin(n_plus_keep_temp,n_max)
                else:
                    a_keep_temp = m - P_n_p*c_keep_temp 
                    n_plus_keep_temp = (1-delta)*n
                    n_plus_keep_temp = np.fmin(n_plus_keep_temp,n_max)

                # ooo. combined policy function
                # i. states
                n_plus_adj_temp = (1-delta)*d_adj_temp
                n_plus_keep_temp = (1-delta)*d_keep_temp

                # ii. policy functions
                d_combined[i_p,i_n,i_b] = P_adj*d_adj_temp + (1-P_adj)*d_keep_temp
                c_combined[i_p,i_n,i_b] = P_adj*c_adj_temp + (1-P_adj)*c_keep_temp
                a_plus[i_p,i_n,i_b] = P_adj*a_adj_temp + (1-P_adj)*a_keep_temp
                n_plus[i_p,i_n,i_b] = P_adj*n_plus_adj_temp + (1-P_adj)*n_plus_keep_temp

                # oooo. value function
                #inv_v[i_p,i_n,i_b] = sigma_eps*np.log(np.exp(inv_v_k/sigma_eps) + np.exp(inv_v_a/sigma_eps))

    return d_combined, c_combined, a_plus, n_plus

@njit
def obj_keep(c,n,m,inv_w,grid_b,d_ubar,sigma,P_n_p): # P_n_p
    """ evaluate bellman equation """

    # a. end-of-period assets
    b = m-P_n_p*c
    
    # b. continuation value
    w = -1.0/linear_interp.interp_1d(grid_b,inv_w,b)

    # c. total value
    value_of_choice = utility(c,n,d_ubar,sigma) + w

    return -value_of_choice # we are minimizing


@njit(parallel=True)
def solve_keep(inv_w,grid_b,grid_m,grid_n,d_ubar,sigma,Np,Nn,Nm,P_n_p): # P_n_p
    """solve bellman equation for keepers using nvfi"""

    # unpack output
    keep_shape = (Np,Nn,Nm)
    inv_v_keep = np.zeros(keep_shape)
    c_keep = np.zeros(keep_shape)

    # loop over outer states
    for i_p in prange(Np):
        for i_n in range(Nn):
            
            # outer states
            n = grid_n[i_n]

            # loop over m state
            for i_m in range(Nm):
                
                # a. cash-on-hand
                m = grid_m[i_m]
                if i_m == 0:
                    c_keep[i_p,i_n,i_m] = 0
                    inv_v_keep[i_p,i_n,i_m] = 0

                # b. optimal choice b = m - c, b <=-0.5.m min 
                c_low = np.fmin(m/2,1e-8)
                c_high = m/P_n_p # m = P_n_p*c (if using everything)
                c_keep[i_p,i_n,i_m] = golden_section_search.optimizer(obj_keep,c_low,c_high,args=(n,m,inv_w[i_p,i_n],grid_b,d_ubar,sigma,P_n_p),tol=1e-8)

                # c. optimal value
                v = -obj_keep(c_keep[i_p,i_n,i_m],n,m,inv_w[i_p,i_n],grid_b,d_ubar,sigma,P_n_p)
                inv_v_keep[i_p,i_n,i_m] = -1.0/v

    return inv_v_keep,c_keep

#######
# adj #
#######

@njit
def obj_adj(d,x,inv_v_keep,grid_n,grid_m,P_d_p):# P_d_p
    """ evaluate bellman equation """

    # a. cash-on-hand
    m = x-P_d_p*d

    # b. durables
    n = d
    
    # c. value-of-choice
    return -linear_interp.interp_2d(grid_n,grid_m,inv_v_keep,n,m)  # we are minimizing

@njit(parallel=True)
def solve_adj(inv_v_keep,c_keep,d_ubar,sigma,n_max,Np,Nx,grid_n,grid_m,grid_x,P_d_p): # P_d_p,
    """solve bellman equation for adjusters using nvfi"""

    # unpack output
    adj_shape = (Np,Nx)
    inv_v_adj = np.zeros(adj_shape)
    d_adj = np.zeros(adj_shape)
    c_adj = np.zeros(adj_shape)

    # loop over outer states
    for i_p in prange(Np):
            
        # loop over x state
        for i_x in range(Nx):
            
            # a. cash-on-hand
            x = grid_x[i_x]
            if i_x == 0:
                d_adj[i_p,i_x] = 0
                c_adj[i_p,i_x] = 0
                inv_v_adj[i_p,i_x] = 0
      
                continue

            # b. optimal choice
            d_low = np.fmin(x/2,1e-8)
            d_high = np.fmin(x/P_d_p,n_max) # cash on hand grid now x = m + (1-tau)*n + B.C. (eg 0.5, or collateral eg. coll_ratio*n)
            d_adj[i_p,i_x] = golden_section_search.optimizer(obj_adj,d_low,d_high,args=(x,inv_v_keep[i_p],grid_n,grid_m,P_d_p),tol=1e-8)

            # c. optimal value
            m = x - P_d_p*d_adj[i_p,i_x]
            c_adj[i_p,i_x] = linear_interp.interp_2d(grid_n,grid_m,c_keep[i_p],d_adj[i_p,i_x],m)
            inv_v_adj[i_p,i_x] = -obj_adj(d_adj[i_p,i_x],x,inv_v_keep[i_p],grid_n,grid_m,P_d_p)

    return inv_v_adj, d_adj, c_adj

def time_iteration(inv_v_keep,inv_v_adj,p_trans,beta,R,sigma,d_ubar,tau,delta,sigma_eps,Np,Nn,Nm,Nx,Nb,grid_p,grid_n,grid_m,grid_x,grid_b,n_max,P_n_p,P_d_p):

    # a. backward iteration
    # i. compute post decision value
    inv_w = compute_w(beta,R,delta,inv_v_keep,inv_v_adj,Np,Nn,Nm,Nb,grid_p,grid_n,grid_m,grid_x,grid_b,p_trans,n_max,tau,sigma_eps,P_d_p)

    # ii. calculate value and policy functions keeper
    inv_v_keep, c_keep = solve_keep(inv_w,grid_b,grid_m,grid_n,d_ubar,sigma,Np,Nn,Nm,P_n_p)

    # iii. calculate value and policy functions adjuster
    inv_v_adj, d_adj, c_adj = solve_adj(inv_v_keep,c_keep,d_ubar,sigma,n_max,Np,Nx,grid_n,grid_m,grid_x,P_d_p)

    # iv. calculate collected policy and value functions using extreme value IID taste shocks
    d_combined, c_combined, b_plus, n_plus = collect_policy(Np,Nn,Nm,Nb,grid_p,grid_n,grid_m,grid_x,grid_b,inv_v_keep,inv_v_adj,c_keep,c_adj,d_adj,sigma_eps,tau,delta,n_max,R,P_d_p,P_n_p)

    return d_combined, c_combined, b_plus, n_plus, inv_v_keep, inv_v_adj