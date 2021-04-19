import numpy as np
from numba import vectorize, njit, prange
import math
import numpy as np
from numba import njit
from quantecon.optimize import root_finding

import utils
from rouwenhorst import rouwenhorst
from het_block import het
from simple_block import simple


'''Part 1: HA block'''

@het(exogenous='Pi_e', policy=['b', 'dg'], backward=['Vb', 'Vd'])  # order as in grid!
def household(Vd_p, Vb_p, Pi_e_p, b_grid, dg_grid, k_grid, e_grid, e_ergodic, sigma_N, sigma_D,
              alpha,delta,r,Div,beta,Ne,Nb,Nd,Nk,P,P_d,P_n,W,N,tau_e,Tax):

    """Single backward iteration step using endogenous gridpoint method for households with separable CRRA utility."""
    sigma = 1.0 # CRRA coefficient
    theta = 0.5 # complements
    chi = 0.581 # weight on non-durables

    # a. grid extensions
    P_n_p = P_n/P
    P_d_p = P_d/P
    tau_a = 0
    lump = (1 + tau_a*(b_grid[np.newaxis,:]/1-1) + tau_e*(e_grid[:,np.newaxis] - 1))*Div
    #if Tax is None: 
    #    Tax = 0
    #z_grid = (1-Tax)*(W/P) * e_grid * N

    z_grid = (W/P) * e_grid * N
    zzz = z_grid[:, np.newaxis, np.newaxis]
    lll = lump[:,:, np.newaxis]

    if Tax is None:
        tax = 0
    else:
        tax = Tax / np.sum(e_ergodic * e_grid) * e_grid
        tax_2d = tax[:,np.newaxis]
        lump -= tax_2d
        lll -= tax_2d[:,:, np.newaxis]

    bbb = b_grid[np.newaxis, :, np.newaxis]
    ddd = dg_grid[np.newaxis, np.newaxis, :]

    # b. pre-compute RHS of combined F.O.C. optimality condition
    rhs = P_d_p + alpha*((dg_grid[:, np.newaxis]/dg_grid[np.newaxis, :]) - (1-delta)) # P_d/P + partial Psi/partial d'

    # c. compute time step 
    Vb,Vd,dg,b,c,c_d = time_iteration(sigma,theta,chi,alpha,delta,r,lump,beta,Pi_e_p,dg_grid,z_grid,b_grid,k_grid,zzz,lll,bbb,ddd,
                   Ne,Nb,Nd,Nk,rhs,Vb_p,Vd_p,P_n_p,P_d_p)

    return Vb, Vd, b, dg, c, c_d

@njit
def optimizer(obj,a,b,args=(),tol=1e-6):
    """ golden section search optimizer
    
    Args:
        obj (callable): 1d function to optimize over
        a (double): minimum of starting bracket
        b (double): maximum of starting bracket
        args (tuple): additional arguments to the objective function
        tol (double,optional): tolerance
    Returns:
        (float): optimization result
    
    """
    
    inv_phi = (np.sqrt(5) - 1) / 2 # 1/phi                                                                                                                
    inv_phi_sq = (3 - np.sqrt(5)) / 2 # 1/phi^2     
        
    # a. distance
    dist = b - a
    if dist <= tol: 
        return (a+b)/2

    # b. number of iterations
    n = int(np.ceil(np.log(tol/dist)/np.log(inv_phi)))

    # c. potential new mid-points
    c = a + inv_phi_sq * dist
    d = a + inv_phi * dist
    yc = obj(c,*args)
    yd = obj(d,*args)

    # d. loop
    for _ in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            dist = inv_phi*dist
            c = a + inv_phi_sq * dist
            yc = obj(c,*args)
        else:
            a = c
            c = d
            yc = yd
            dist = inv_phi*dist
            d = a + inv_phi * dist
            yd = obj(d,*args)

    # e. return
    if yc < yd:
        return (a+d)/2
    else:
        return (c+b)/2

@njit(fastmath=True) 
def Psi_fun(alpha,delta,dp,d):
    ''' durable adjustment cost function '''
    return 0.5*alpha*((dp-(1-delta)*d)/d)**2*d

@njit(fastmath=True) 
def euler_obj(c,d,chi,theta,sigma,rhs):
    ''' euler objective for finding c '''
    
    if c < 1e-8:
        c = 1e-8

    uc_1 = chi**(1/theta)*c**(-1/theta)* (chi**(1/theta)*c**((theta-1)/theta) + (1-chi)**(1/theta)*d**((theta-1)/theta) )**(1/(theta-1)) 
    uc_2 = (chi**(1/theta)*c**((theta-1)/theta) + (1-chi)**(1/theta)*d**((theta-1)/theta) )**(-sigma*theta/(theta-1))
    uc = uc_1*uc_2

    res = uc-rhs # residual

    return res**2 # least-squares approach to finding root

@njit(fastmath=True,parallel=True)
def solve_unconstrained(sigma,theta,chi,alpha,delta,r,d_grid,b_grid,
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
                
                rhs = Wb_endo[ie, ib, i_d]
                d = d_grid[i_d]
                res = optimizer(euler_obj, a=1e-8,b=15, args=(d,chi,theta,sigma,rhs))
                c[ie,ib,i_d] = res
                # use consav newton, guess 1
                #c[ie, ib, i_d] = theta**(-1)*Wb_endo[ie, ib, i_d] * (d_grid[i_d])**((theta-1) * (1-sigma))
                if c[ie, ib, i_d] <= 0:
                    c[ie, ib, i_d] = 1e-8 # for numerical stability
                
                #c[ie, ib, i_d] **= (1/(theta*(1-sigma)-1))

                # d. find savings b(e,b',d) from B.C.
                b_endo_unc[ie, ib, i_d] = (P_n_p*c[ie, ib, i_d] + P_d_p*(dp_endo[ie, ib, i_d] - (1 - delta) * d_grid[i_d]) + b_grid[ib] + Psi_fun(alpha,delta,dp_endo[ie, ib, i_d],d_grid[i_d]) -
                            z_grid[ie] - lump[ie,ib]) / (1 + r)
    
    return dp_endo, Wb_endo, b_endo_unc, c

@njit(fastmath=True,parallel=True)
def solve_constrained(sigma,theta,chi,alpha,delta,r,d_grid,b_grid,k_grid,
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

                rhs = Wb_endo[ie, ik, i_d]
                d = d_grid[i_d]
                res = optimizer(euler_obj, a=1e-8,b=15, args=(d,chi,theta,sigma,rhs))
                c[ie,ik,i_d] = res
                #c[ie, ik, i_d] = theta**(-1)*Wb_endo[ie, ik, i_d] * (d_grid[i_d])**((theta-1) * (1-sigma))
                if c[ie, ik, i_d] <= 0:
                    c[ie, ik, i_d] = 1e-8 # for numerical stability
                
                #c[ie, ik, i_d] **= (1/(theta*(1-sigma)-1))

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

def time_iteration(sigma,theta,chi,alpha,delta,r,lump,beta,Pi_e,d_grid,z_grid,b_grid,k_grid,zzz,lll,bbb,ddd,
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
    dp_endo_unc, _, b_endo_unc, _ = solve_unconstrained(sigma,theta,chi,alpha,delta,r,d_grid,b_grid,
                        z_grid,lump,Ne,Nb,Nd,lhs_unc,rhs,Wb,P_d_p,P_n_p)
    # ii. use d'(z,b',d), b(z,b',d) to interpolate to d'(z,b,d), b'(z,b,d)
    i, pi = utils.interpolate_coord(b_endo_unc.swapaxes(1, 2), b_grid) # gives interpolation weights from a' grid to a grid
    d_unc[:,:,:] = utils.apply_coord(i, pi, dp_endo_unc.swapaxes(1, 2)).swapaxes(1, 2) # apply weights to d'(z,a',d)->d'(z,a,d)
    b_unc[:,:,:] = utils.apply_coord(i, pi, b_grid).swapaxes(1, 2) # apply weights to  a'(z,a',d)->a'(z,a,d)

    # c. get constrained solution to d'(z,0,d), b'(z,0,d), c(z,0,d)
    # i. get d'(z,k,d), b(z,k,d), c(z,k,d) with EGM
    dp_endo_con, _, b_endo_con, _ = solve_constrained(sigma,theta,chi,alpha,delta,r,d_grid,b_grid,k_grid,
                      z_grid,lump,Ne,Nk,Nd,lhs_con,rhs,Wb,P_n_p,P_d_p)
    # ii. get d'(z,b,d) by interpolating using b'(z,k,d)
    d_con[:] = utils.interpolate_y(b_endo_con[:, ::-1, :].swapaxes(1, 2), b_grid, 
                                       dp_endo_con[:, ::-1, :].swapaxes(1, 2)).swapaxes(1, 2)

    # d. collect policy functions d',b',c by combining unconstrained and constrained solutions
    d, b, c = collect_policy(delta,alpha,r,Ne,Nb,Nd,b_grid,zzz,lll,ddd,bbb,d_unc,b_unc,d_con,P_n_p,P_d_p)
    c_d = d - (1-delta)*ddd # durable investment policy fuctions

    # e. update Vb, Vd
    # i. compute marginal utilities
    c[c<0] = 1e-8 # for numerical stability while converging
    uc_1 = chi**(1/theta)*c**(-1/theta)* (chi**(1/theta)*c**((theta-1)/theta) + (1-chi)**(1/theta)*ddd**((theta-1)/theta) )**(1/(theta-1)) 
    uc_2 = (chi**(1/theta)*c**((theta-1)/theta) + (1-chi)**(1/theta)*ddd**((theta-1)/theta) )**(-sigma*theta/(theta-1)) 
    uc = uc_1*uc_2
    ud_1 = (1-chi)**(1/theta)*ddd**(-1/theta)* (chi**(1/theta)*c**((theta-1)/theta) + (1-chi)**(theta/theta)*ddd**((theta-1)/theta) )**(1/(theta-1)) 
    ud_2 = (chi**(1/theta)*c**((theta-1)/theta) + (1-chi)**(1/theta)*ddd**((theta-1)/theta) )**(-sigma*theta/(theta-1)) 
    ud = ud_1*ud_2

    # ii. compute Vb, Vd using envelope conditions
    Vb = (1/P_n_p)*(1 + r) * uc
    Vd = ud + (1/P_n_p)*uc * (P_d_p*(1-delta) - 0.5*alpha*((1-delta)**2 - (d/ddd)**2))

    return Vb,Vd,d,b,c,c_d
