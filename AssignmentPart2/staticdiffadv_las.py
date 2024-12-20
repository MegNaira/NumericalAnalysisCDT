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

#initializing diffusion coefficient
D=1e+5

#initializing wind velocity and direction
u_dir={
       'N' : -np.array([0,1])*10,
       'R' : -np.array([0.49,0.87])*10
       }

#initializing lengthscales
Ls = {
    '1_25' : 1.25,
    '2_5' : 2.5,
    '5' : 5,
    '10' : 10,
    '20' : 20,
    '40' : 40
    }

BCs = {
       'whole' : True,
       'South': False
       }


#initialiazing gaussian fire source
def source(X):
    """
    Computes the value of a 2D Gaussian function at given points, centered at the Mountbatten Building,
    or returns the identity if the gauss flag is deactivated.

    Parameters
    ----------
    X : array of floats 1x2
        global coordinates

    Returns
    -------
    Gaussian values at the specified (x, y) coordinates.
    
    """
    x0=442365.
    y0=115483.
    sigma_x=500.
    sigma_y=500.
    amplitude=1.

    return amplitude * np.exp(-(((X[0] - x0)**2) / (2 * sigma_x**2) + ((X[1] - y0)**2) / (2 * sigma_y**2)))


#solver that uses functions defined in the imported lib
def staticdiffadvsolver(v, D, res, BC, plot=True):
    """
    Solves the static advection diffusion equation

    Parameters
    ----------
    v : string
        either 'R' (Reading) or 'N' (North) depending on the wind direction
        
    D : float
        diffusion coefficient, should be of the order of 10**4 |u|, for the resolutions considered
        
    source : func 1d
            function to use as fire source function. Here either source_gauss or source_id.
        
    res : string from ['1_25','2_5','5','10','20','40']
        typical lengthscale (in km) of the grid considered.
        
    BC : boolean
        If True, sets homogeneous Dirichlet BCs on all boundary nodes.
        If False, only on boundary nodes whose y<ysoton.
        
    plot : boolean, optional
        parameter that activates the map plot if True. The default is True.

    Returns
    -------
    N_equations : float
        Number of equations used to solve at the given resolution, i.e. degrees of freedom of the problem
    Psi_R_norm : float
        pollutant concentration in Reading normalized to 1 over Southampton

    """
    
    nodes = np.loadtxt('las_grids/las_nodes_%sk.txt'%res)
    IEN = np.loadtxt('las_grids/las_ien_%sk.txt'%res, dtype=np.int64)
    #IEN = IEN[:,-1::-1]
    boundary_nodes = np.loadtxt('las_grids/las_bdry_%sk.txt'%res, dtype=np.int64)
    
    #setting homogeneous Dirichlet BCs on boundary nodes
    bkey=[key for key, value in BCs.items() if value == BC][0]
    if (BC):
        bdry=boundary_nodes
    else:
        bdry=[0]
        for i in boundary_nodes[1:]:
            if (nodes[i,1]<110000) and ((nodes[i,1]-115483)<=500): #y<ysoton
                bdry = np.append(bdry,boundary_nodes[i])
        
    ID = np.zeros(len(nodes), dtype=np.int64)
        
    n_eq = 0
    for i in range(len(nodes[:, 1])):
        if i in bdry:
            ID[i] = -1
        else:
            ID[i] = n_eq
            n_eq += 1
    
    N_equations = np.max(ID)+1
    N_elements = IEN.shape[0]
    N_nodes = nodes.shape[0]
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
        k_e = lib.local_stiffness(nodes[IEN[e,:],:], u_dir[v], D)
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
    Rx=np.array([473993, 171625])
    psi_R = lib.psi_p(nodes, IEN, Psi_A, Rx)
    Sx=np.array([442365, 115483])
    psi_S = lib.psi_p(nodes, IEN, Psi_A, Sx)
    
    Psi_R_norm = psi_R/psi_S
    
    #map plot
    if (plot):
    
        plt.cla()
        plt.tripcolor(nodes[:,0], nodes[:,1], Psi_A, triangles=IEN)
        
        plt.scatter([473993],[171625], s=2, c='red')
        plt.scatter(nodes[bdry,0],nodes[bdry,1], s=2, c='black')
        
        plt.xlabel(r"longitude", fontsize=12)
        plt.ylabel(r"latitude", fontsize=12)
        plt.title('Static advection diffusion with u = 10 m/s %s and D=%.0e'%(v,D), fontsize=8)
        plt.text(330000,250000,'L = %.2f km'%Ls[res], fontsize=6)
        plt.text(460000,175000,'$\psi$ = %.3f'%Psi_R_norm, fontsize=6, c='red')
        
        plt.tick_params(labelleft=False, labelbottom=False)
        plt.colorbar()
        
        plt.savefig('StatAdvDiff_uto%s_B%s_%s.pdf'%(v,bkey,res), format="pdf", bbox_inches="tight")
        plt.show()
        
        print("The pollutant concentration over Reading, normalized to one over Southampton, is %.3f. \n"%Psi_R_norm)
    
    return N_equations, Psi_R_norm

##%% 
def accuracy(v, D, mapres, BC):
    """
    Gives result and colormap for a desired resolution (mapres),
    Shows the convergence of the solver from a fit analysis and from a theoretical computation of slope and error.
    

    Parameters
    ----------    
    v : string
        either 'R' (Reading) or 'N' (North) depending on the wind direction
        
    D : float
        diffusion coefficient, should be of the order of 10**4 |u|, for the resolutions considered
        
    mapres : string from ['1_25','2_5','5','10','20','40']
        resolution considered for the colormap plot
        
    BC : boolean
        If True, sets homogeneous Dirichlet BCs on all boundary nodes.
        If False, only on boundary nodes whose y<ysoton.

    Returns
    -------
    None

    """
    
    #keyword for boundary conditions
    bkey=[key for key, value in BCs.items() if value == BC][0]
    
    #initializing storage for solutions
    Psi_R = []
    DOF = []

    for l in Ls:
        
        if l == mapres:
        
            dof, Psi = staticdiffadvsolver(v, D, l, BC)
            
        else:
            
            dof, Psi = staticdiffadvsolver(v, D, l, BC, plot=False)
        
        Psi_R.append(Psi)
        DOF.append(dof)
    
    #taking, in order, worst resolutions, mid resolutions, and best resolutions    
    y_4N=np.array([Psi_R[3],Psi_R[1],Psi_R[0]])
    y_2N=np.array([Psi_R[4],Psi_R[2],Psi_R[1]])
    y_N=np.array([Psi_R[5],Psi_R[3],Psi_R[2]])
    
    s=np.zeros(len(y_4N))
    error=np.zeros(len(y_4N))
    
    for k in range(len(y_4N)):
        s[k]=np.log2(abs((y_2N[k]-y_N[k])/(y_4N[k]-y_2N[k])))
        error[k]=abs(y_2N[k]-y_N[k])/(1-2**(-s[k]))
    
    np.set_printoptions(precision=2)
    print("\nTheoretical convergence slopes:")    
    print('s = ',s,' from worst 3 to best 3 resolutions')
    print('mean s = %.2f'%np.mean(s))
    print("\nTheoretical computed errors:")
    print('error = ',error,' from worst 3 to best 3 resolutions')
    print('mean error = %.2f'%np.mean(error))
    print('\n')
    
    #setting up array for relative errors
    err_wrt_best=np.zeros(6)
    
    for i in range(len(err_wrt_best)):
        err_wrt_best[i]=abs(Psi_R[i]-Psi_R[0])
    
    #finding slope of convergence with fit
    polyfit_coeffs = np.polyfit(np.log(DOF[1:4]),np.log(err_wrt_best[1:4]),1) 

    trendline = lambda coeff,x: np.poly1d(coeff)(x)
    
    plt.figure()
    plt.loglog(DOF[1:4], err_wrt_best[1:4], 'xk')
    plt.loglog(DOF[4:], err_wrt_best[4:], 'xb')
    plt.loglog(DOF, np.exp(trendline(polyfit_coeffs,np.log(DOF))),'-r', label=rf'$\propto N^{{{polyfit_coeffs[0]:.2f}}}$')
    plt.xlabel('Degrees of freedom')
    plt.ylabel('Error relative to best solution')
    plt.legend()
    plt.grid()
    plt.title('Relative error for u=10 m/s %s, D=%.0e, static, bdry=%s'%(v,D,bkey))
    plt.savefig('ConvergenceStatAdvDiff_uto%s_B%s.pdf'%(v,bkey), format="pdf", bbox_inches="tight")
    
    plt.show()


#using the defined functions to get to results for different directions of the velocity and different BCs
mapres= '1_25'
print("The grid considered for this computation has typical lengthscale of %.2f km.\n"%Ls[mapres])
for v in u_dir:
    
    print("The wind is 10 m/s %s, the diffusion coefficient is %.0e and the source is Gaussian."%(v,D))
    
    for b in BCs:
        print("Homogeneous Dirichlet BCs are now set on the %s boundary."%b)
        
        #analysis
        accuracy(v, D, mapres, BC=BCs[b])
        