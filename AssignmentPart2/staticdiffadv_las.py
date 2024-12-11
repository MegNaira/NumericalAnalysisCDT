#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:48:44 2024

@author: arianna
"""

import matplotlib.pyplot as plt
import numpy as np
import librarystatic2d_wADV as lib
from scipy import sparse as sp

#initializing wind velocity and diffusion coefficient
utoR=-np.array([0.49,0.87])*10
utoN=-np.array([0,1])*10
D=10000
#initializing desidered resolution
L='1_25'

#initialiazing gaussian source
def source(X):
    """
    Compute the value of a 2D Gaussian function at given points, centered at the Mountbatten Building.

    Parameters:
    - X: array-like 2x1, coordinates.
    

    Returns:
    - Gaussian values at the specified (x, y) coordinates.
    """
    x0=442365.
    y0=115483.
    sigma_x=500.
    sigma_y=500.
    amplitude=1.
    return amplitude * np.exp(-(((X[0] - x0)**2) / (2 * sigma_x**2) + ((X[1] - y0)**2) / (2 * sigma_y**2)))

#solver that uses functions defined in the imported lib
def staticdiffadvsolver(resolution,u,D,plot=True):
    """
    Solves the static advection diffusion equation

    Parameters
    ----------
    resolution : string from ['1_25','2_5','5','10','20','40']
        typical lengthscale (in km) of the grid considered.
    u : array of floats, 2x1
        wind velocity in the xy plane
    
    D : float
        diffusion coefficient, should be of the order of 10**5 |u|, for the resolutions considered
    plot : boolean, optional
        parameter that activates the map plot if True. The default is True.

    Returns
    -------
    N_equations : float
        Number of equations used to solve at the given resolution, i.e. degrees of freedom of the problem
    Psi_R_norm : float
        pollutant concentration in Reading normalized to 1 over Southampton

    """

    nodes = np.loadtxt('las_grids/las_nodes_%sk.txt'%resolution)
    IEN = np.loadtxt('las_grids/las_ien_%sk.txt'%resolution, dtype=np.int64)
    boundary_nodes = np.loadtxt('las_grids/las_bdry_%sk.txt'%resolution, dtype=np.int64)
    
    #setting homogeneous Dirichlet BCs on boundary nodes whose y<y_soton
    lower_bdry=[0]
    for i in boundary_nodes[1:]:
        if (nodes[i,1]<110000) and ((nodes[i,1]-115483)<=50):
            lower_bdry = np.append(lower_bdry,boundary_nodes[i])
        
    ID = np.zeros(len(nodes), dtype=np.int64)
        
    n_eq = 0
    for i in range(len(nodes[:, 1])):
        if i in lower_bdry:
            ID[i] = -1
        else:
            ID[i] = n_eq
            n_eq += 1
    
    N_equations = np.max(ID)+1
    N_elements = IEN.shape[0]
    N_nodes = nodes.shape[0]
    N_dim = nodes.shape[1]
    # Location matrix
    LM = np.zeros_like(IEN.T)
    for e in range(N_elements):
        for a in range(3):
            LM[a,e] = ID[IEN[e,a]]
    # Global stiffness matrix and force vector
    K = np.zeros((N_equations, N_equations))
    F = np.zeros((N_equations,))
    # Loop over elements
    for e in range(N_elements):
        k_e = lib.local_stiffness(nodes[IEN[e,:],:], u, D)
        f_e = lib.local_force(source, nodes[IEN[e,:],:])
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    K[A, B] += k_e[a, b]
            if (A >= 0):
                F[A] += f_e[a]
    # Solve
    K = sp.csr_matrix(K)
    Psi_interior = sp.linalg.spsolve(K, F)
    Psi_A = np.zeros(N_nodes)
    for n in range(N_nodes):
        if ID[n] >= 0: # Otherwise Psi should be zero, and we've initialized that already.
            Psi_A[n] = Psi_interior[ID[n]]
    
    #computation of the target quantities
    psi_R = lib.psi_r(nodes, IEN, Psi_A)
    psi_S = lib.psi_s(nodes, IEN, Psi_A)
    
    Psi_R_norm = psi_R/psi_S
    
    #map plot
    if (plot):
    
        plt.cla()
        plt.tripcolor(nodes[:,0], nodes[:,1], Psi_A, triangles=IEN)
        
        plt.scatter([473993],[171625], s=2, c='red')
        #plt.scatter([442365,473993],[115483,171625], s=2, c='red')
        plt.scatter(nodes[lower_bdry,0],nodes[lower_bdry,1], s=2)
        #plt.scatter(nodes[:,0],nodes[:,1], s=1)
        
        plt.xlabel(r"longitude", fontsize=12)
        plt.ylabel(r"latitude", fontsize=12)
        plt.title('Static advection diffusion with u = 10 km/s R-directed and D=%i'%D, fontsize=8)
        plt.text(330000,250000,'L = 1.25 km', fontsize=6)
        plt.text(460000,175000,'$\psi$ = %.3f'%Psi_R_norm, fontsize=6, c='red')
        
        plt.tick_params(labelleft=False,labelbottom=False)
        plt.colorbar()
        plt.show()
        plt.savefig('StatAdvDiff_utoR_%s.pdf'%resolution)
    
    return N_equations, Psi_R_norm


DOF1_25, Psi1_25 = staticdiffadvsolver(L, utoR, D)
#results
print("The grid considered for this computation has typical lengthscale of %s km."%L)
print("The wind is 10 m/s towards North and the diffusion coefficient is %i."%D)
print("The pollutant concentration over Reading, normalized to one over Southampton, is %.3f."%Psi1_25)

#%% 
def accuracy(maxDOF,bestPsi):
    """
    Shows the convergence of the solver from a fit analysis and from a theoretical computation of slope and error.
    

    Parameters
    ----------
    maxDOF : float
        Degrees of freedom from the best resolution available
    bestPsi : float
        Solution from the solver for the best resolution available

    Returns
    -------
    None.

    """
    
    #initializing more lengthscales
    Ls=['2_5','5','10','20','40']
    
    #storing the max resolution solution
    Psi_R = [bestPsi]
    DOF = [maxDOF]

    for l in Ls:
        
        dof, Psi = staticdiffadvsolver(l, utoR, D, plot=False)
        
        Psi_R.append(Psi)
        DOF.append(dof)
    
    #taking, in order, worst resolutions, mid resolutions, and best resolutions    
    y_4N=np.array([Psi_R[3],Psi_R[1],Psi_R[2],Psi_R[0]])
    y_2N=np.array([Psi_R[4],Psi_R[2],Psi_R[3],Psi_R[1]])
    y_N=np.array([Psi_R[5],Psi_R[3],Psi_R[4],Psi_R[2]])
    
    s=np.zeros(len(y_4N))
    error=np.zeros(len(y_4N))
    
    for k in range(len(y_4N)):
        s[k]=-np.log2(abs((y_2N[k]-y_N[k])/(y_4N[k]-y_2N[k])))
        error[k]=abs(y_2N[k]-y_N[k])/(1-2**(-s[k]))
        
    print('s=',s)
    print('error=',error)
    print('mean s=',np.mean(s))
    print('mean error=',np.mean(error))
    
    #setting up array for relative errors
    err_wrt_best=np.zeros(6)
    
    for i in range(len(err_wrt_best)):
        err_wrt_best[i]=abs(Psi_R[i]-Psi_R[0])
    
    #finding slope of convergence with fit
    polyfit_coeffs = np.polyfit(np.log(DOF[1:4]),np.log(err_wrt_best[1:4]),1) 

    trendline = lambda data,x: np.poly1d(data)(x)
    
    plt.figure()
    plt.loglog(DOF, err_wrt_best, 'xk')
    plt.loglog(DOF, np.exp(trendline(polyfit_coeffs,np.log(DOF))),'-r', 
               label=rf'$\propto N^{{{polyfit_coeffs[0]:.2f}}}$')
    plt.xlabel('Degrees of freedom')
    plt.ylabel('Error relative to max res sol')
    plt.legend()
    plt.grid()
    plt.title(f'Relative Error for u=10 km/s R, D=100000, static')
    plt.savefig('ConvergenceStatAdvDiff_utoR.pdf')
    
    plt.show()   

accuracy(DOF1_25,Psi1_25)       