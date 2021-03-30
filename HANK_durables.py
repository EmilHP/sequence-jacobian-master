import numpy as np
from numba import vectorize, njit, prange

import utils
from rouwenhorst import rouwenhorst
from het_block import het
from simple_block import simple


'''Part 1: HA block'''

@het(exogenous='Pi_e', policy=['b', 'dg'], backward=['Vb', 'Vd'])  # order as in grid!
def household(Vd_p, Vb_p, Pi_e_p, b_grid, dg_grid, k_grid, e_grid, sigma_N, sigma_D,
              alpha,delta,r,Div,beta,Ne,Nb,Nd,Nk,P,P_d,P_n,W,N,tau_e):

    """Single backward iteration step using endogenous gridpoint method for households with separable CRRA utility."""

    # a. grid extensions
    P_n_p = P_n/P
    P_d_p = P_d/P
    tau_a = 0
    lump = (1 + tau_a*(b_grid[np.newaxis,:]/1-1) + tau_e*(e_grid[:,np.newaxis] - 1))*Div
    z_grid = (W/P_n) * e_grid * N
    zzz = z_grid[:, np.newaxis, np.newaxis]
    lll = lump[:,:, np.newaxis]
    bbb = b_grid[np.newaxis, :, np.newaxis]
    ddd = dg_grid[np.newaxis, np.newaxis, :]

    # b. pre-compute RHS of combined F.O.C. optimality condition
    rhs = P_d_p + alpha*((dg_grid[:, np.newaxis]/dg_grid[np.newaxis, :]) - (1-delta)) # P_d/P + partial Psi/partial d'

    # c. compute time step 
    Vb,Vd,dg,b,c = time_iteration(sigma_N,sigma_D,alpha,delta,r,lump,beta,Pi_e_p,dg_grid,z_grid,b_grid,k_grid,zzz,lll,bbb,ddd,
                   Ne,Nb,Nd,Nk,rhs,Vb_p,Vd_p,P_n_p,P_d_p)

    return Vb, Vd, b, dg, c

@njit(fastmath=True) 
def Psi_fun(alpha,delta,dp,d):
    ''' durable adjustment cost function '''
    return 0.5*alpha*((dp-(1-delta)*d)/d)**2*d

@njit(fastmath=True,parallel=True)
def solve_unconstrained(sigma_N,alpha,delta,r,d_grid,b_grid,
                        z_grid,lump,Ne,Nb,Nd,lhs_unc,rhs,Wb,P_d_p,P_n_p):

    ''' Finds next period d, b and current periods c in the unconstrained case '''
    
    # output containers
    sol_shape = (Ne, Nb, Nd)
    dp_endo = np.zeros(sol_shape)
    Wb_endo = np.zeros(sol_shape)
    b_endo_unc = np.zeros(sol_shape)
    c = np.zeros(sol_shape)

    # a. EGM loop
    for ie in range(Ne): # loop over e (state)
        for ibp in range(Nb): # loop over b' (state)
            idp = 0.0  # use mononicity in d
            for i_d in range(Nd): # loop over d (state)
                while True: # loop over d' (choice)
                    if lhs_unc[ie, ibp, int(idp)] < rhs[int(idp), i_d]: # stop before corner solution in d'
                        break
                    elif int(idp) < Nd - 1: # below maxmimum d (according to grid)
                        idp += 1
                    else:
                        break
                if idp == 0.0: # i. lower grid bound, corner solution in d'
                    dp_endo[ie, ibp, i_d] = 0.001
                    Wb_endo[ie, ibp, i_d] = Wb[ie, ibp, 0]
                elif idp == Nd: # ii. upper grid bound
                    dp_endo[ie, ibp, i_d] = d_grid[int(idp)]
                    Wb_endo[ie, ibp, i_d] = Wb[ie, ibp, int(idp)]
                else: # iii. inner solution
                    # o. move around grid values to find interpolation weights for finding endogenous value of d'
                    y0 = lhs_unc[ie, ibp, int(idp) - 1] - rhs[int(idp) - 1, i_d] 
                    y1 = lhs_unc[ie, ibp, int(idp)] - rhs[int(idp), i_d]
                    # oo. apply weights to interpolate to d'(e,b',d), linear interpolation where LHS is x, RHS is y and d is grid to interpolate to
                    dp_endo[ie, ibp, i_d] = d_grid[int(idp) - 1] - y0 * (d_grid[int(idp)] - d_grid[int(idp) - 1]) / (y1 - y0)
                    # ooo. interpolate from Wb(e,b',d') -> Wb(e,b',d) using d'(e,b',d)
                    Wb_endo[ie, ibp, i_d] = Wb[ie, ibp, int(idp) - 1] + (dp_endo[ie, ibp, i_d] - d_grid[int(idp) - 1]) * \
                                            (Wb[ie, ibp, int(idp)] - Wb[ie, ibp, int(idp) - 1]) / (d_grid[int(idp)] \
                                            - d_grid[int(idp) - 1])
                
    # c. find consumption, c(e,b',d) from F.O.C. wrt. b'
    for ie in prange(Ne): # parallel
        for ib in range(Nb):
            for i_d in range(Nd):
                c[ie, ib, i_d] = Wb_endo[ie, ib, i_d]
                if c[ie, ib, i_d] <= 0:
                    c[ie, ib, i_d] = 1e-8 # for numerical stability
                
                c[ie, ib, i_d] **= -1/sigma_N

                # d. find savings b(e,b',d) from B.C.
                b_endo_unc[ie, ib, i_d] = (P_n_p*c[ie, ib, i_d] + P_d_p*(dp_endo[ie, ib, i_d] - (1 - delta) * d_grid[i_d]) + b_grid[ib] + Psi_fun(alpha,delta,dp_endo[ie, ib, i_d],d_grid[i_d]) -
                            z_grid[ie] - lump[ie,ib]) / (1 + r)
    
    return dp_endo, Wb_endo, b_endo_unc, c

@njit(fastmath=True,parallel=True)
def solve_constrained(sigma_N,alpha,delta,r,d_grid,b_grid,k_grid,
                      z_grid,lump,Ne,Nk,Nd,lhs_con,rhs,Wb,P_n_p,P_d_p):

    ''' Finds next period d, b and current periods c in the constrained case '''

    # output containers
    sol_shape = (Ne, Nk, Nd)
    dp_endo = np.zeros(sol_shape)
    Wb_endo = np.zeros(sol_shape)
    b_endo_con = np.zeros(sol_shape)
    c = np.zeros(sol_shape)

    # a. EGM loop
    for ie in range(Ne): # loop over e
        for ik in range(Nk): # loop over liquidity constraint multiplier
            idp = 0  # use mononicity in d
            for i_d in range(Nd): # loop over d
                while True: # loop over d'
                    if lhs_con[ie, ik, int(idp)] < rhs[int(idp), i_d]: # stop before corner solution in d' (lambda>0)
                        break
                    elif idp < Nd - 1:
                        idp += 1
                    else:
                        break
                if idp == 0: # i. lower grid bound, if idp == 0: go to corner solution in d'
                    dp_endo[ie, ik, i_d] = 0.001
                    Wb_endo[ie, ik, i_d] = (1 + k_grid[ik]) * Wb[ie, 0, 0]
                elif idp == Nd: # ii. upper grid bound
                    dp_endo[ie, ik, i_d] = d_grid[int(idp)]
                    Wb_endo[ie, ik, i_d] = (1 + k_grid[ik]) * Wb[ie, 0, int(idp)]
                else: # iii. inner solution
                    # o. move around grid values to find interpolation weights for finding endogenous value of d'
                    y0 = lhs_con[ie, ik, int(idp) - 1] - rhs[int(idp) - 1, i_d]
                    y1 = lhs_con[ie, ik, int(idp)] - rhs[int(idp), i_d]
                    # oo. apply weights to interpolate to d'(e,k,d), linear interpolation where LHS is x, RHS is y and d is grid to interpolate to
                    dp_endo[ie, ik, i_d] = d_grid[int(idp) - 1] - y0 * (d_grid[int(idp)] - d_grid[int(idp) - 1]) / (y1 - y0)
                    # ooo. interpolate from Wb(e,0,d) -> -(1+k)*Wb(e,k,d) using d'(e,k,d)
                    Wb_endo[ie, ik, i_d] = (1 + k_grid[ik]) * (
                            Wb[ie, 0, int(idp) - 1] + (dp_endo[ie, ik, i_d] - d_grid[int(idp) - 1]) *
                            (Wb[ie, 0, int(idp)] - Wb[ie, 0, int(idp) - 1]) / (d_grid[int(idp)] - d_grid[int(idp) - 1]))

    # c. find consumption, c(e,k,d) from F.O.C. wrt. b'
    for ie in prange(Ne): # parallel
        for ik in range(Nk):
            for i_d in range(Nd):
                c[ie, ik, i_d] = Wb_endo[ie, ik, i_d]
                if c[ie, ik, i_d] <= 0:
                    c[ie, ik, i_d] = 1e-8 # for numerical stability
                
                c[ie, ik, i_d] **= -(1/sigma_N)

                # d. find savings b(e,k,d) from B.C.
                b_endo_con[ie, ik, i_d] = (P_n_p*c[ie, ik, i_d] + P_d_p*(dp_endo[ie, ik, i_d] - (1 - delta) * d_grid[i_d]) + b_grid[0] + Psi_fun(alpha,delta,dp_endo[ie, ik, i_d],d_grid[i_d]) -
                            z_grid[ie] - lump[ie,0]) / (1 + r)
    
    return dp_endo, Wb_endo, b_endo_con, c

def collect_policy(delta,alpha,r,Ne,Nb,Nd,b_grid,zzz,lll,ddd,bbb,d_unc,b_unc,d_con,P_n_p,P_d_p):

    ''' collects policy functions from constrained and unconstrained solution '''
    
    # output containers
    sol_shape = (Ne, Nb, Nd)
    d = np.zeros(sol_shape)
    b = np.zeros(sol_shape)
    c = np.zeros(sol_shape)

    # a. start with unconstrained solution and apply constraint to d', n' solution
    d[:], b[:] = d_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0] # if lower values than constraint, impose b_min
    d[b <= b_grid[0]] = d_con[b <= b_grid[0]] # if lower values in assets than constraint, use d_con solution
    # b. use B.C. to obtain c solution
    c[:] = (zzz + lll - P_d_p*(d - (1 - delta) * ddd ) + (1 + r) * bbb - Psi_fun(alpha,delta, d, ddd)  - b)/P_n_p
    
    return d, b, c

def time_iteration(sigma_N,sigma_D,alpha,delta,r,lump,beta,Pi_e,d_grid,z_grid,b_grid,k_grid,zzz,lll,bbb,ddd,
                   Ne,Nb,Nd,Nk,rhs,Vb,Vd,P_n_p,P_d_p):
    
    # output containers
    sol_shape = (Ne, Nb, Nd)
    Wb = np.zeros(sol_shape)
    Wd = np.zeros(sol_shape)
    
    d_unc = np.zeros(sol_shape)
    b_unc = np.zeros(sol_shape)
    d_con = np.zeros(sol_shape)

    ''' computes one backward EGM step '''

    # a. pre-compute post-decision value functions and LHS offirst-order optimality equation
    # i. post-decision functions
    Wb[:,:,:] = (Vb.T @ (beta * Pi_e.T)).T
    Wd[:,:,:] = (Vd.T @ (beta * Pi_e.T)).T
    # ii. LHS of FOC optimality equation in unconstrained and constrained case
    lhs_unc = Wd / Wb
    lhs_con = lhs_unc[:, 0, :]
    lhs_con = lhs_con[:, np.newaxis, :] / (1 + k_grid[np.newaxis, :, np.newaxis]) 

    # b. get unconstrained solution to d'(z,b,d), a'(z,b,d), c(z,b,d)
    # i. get d'(z,b',d), b(z,b',d), c(z,b,d) with EGM
    dp_endo_unc, _, b_endo_unc, _ = solve_unconstrained(sigma_N,alpha,delta,r,d_grid,b_grid,
                        z_grid,lump,Ne,Nb,Nd,lhs_unc,rhs,Wb,P_d_p,P_n_p)
    # ii. use d'(z,b',d), b(z,b',d) to interpolate to d'(z,b,d), b'(z,b,d)
    i, pi = utils.interpolate_coord(b_endo_unc.swapaxes(1, 2), b_grid) # gives interpolation weights from a' grid to a grid
    d_unc[:,:,:] = utils.apply_coord(i, pi, dp_endo_unc.swapaxes(1, 2)).swapaxes(1, 2) # apply weights to d'(z,a',d)->d'(z,a,d)
    b_unc[:,:,:] = utils.apply_coord(i, pi, b_grid).swapaxes(1, 2) # apply weights to  a'(z,a',d)->a'(z,a,d)

    # c. get constrained solution to d'(z,0,d), b'(z,0,d), c(z,0,d)
    # i. get d'(z,k,d), b(z,k,d), c(z,k,d) with EGM
    dp_endo_con, _, b_endo_con, _ = solve_constrained(sigma_N,alpha,delta,r,d_grid,b_grid,k_grid,
                      z_grid,lump,Ne,Nk,Nd,lhs_con,rhs,Wb,P_n_p,P_d_p)
    # ii. get d'(z,b,d) by interpolating using b'(z,k,d)
    d_con[:] = utils.interpolate_y(b_endo_con[:, ::-1, :].swapaxes(1, 2), b_grid, 
                                       dp_endo_con[:, ::-1, :].swapaxes(1, 2)).swapaxes(1, 2)

    # d. collect policy functions d',b',c by combining unconstrained and constrained solutions
    d, b, c = collect_policy(delta,alpha,r,Ne,Nb,Nd,b_grid,zzz,lll,ddd,bbb,d_unc,b_unc,d_con,P_n_p,P_d_p)

    # e. update Vb, Vd
    # i. compute marginal utilities
    c[c<0] = 1e-8 # for numerical stability while converging
    uc = c**(-1/sigma_N)
    ud = (ddd)**(-1/sigma_D)
    # ii. compute Vb, Vd using envelope conditions
    Vb = (1/P_n_p)*(1 + r) * uc
    Vd = ud + (1/P_n_p)*uc * (P_d_p*(1-delta) - 0.5*alpha*((1-delta)**2 - (d/ddd)**2))

    return Vb,Vd,d,b,c
