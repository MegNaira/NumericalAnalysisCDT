#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:10:22 2024

@author: arianna
"""

import matplotlib.pyplot as plt
import numpy as np
import librarystatic2d as lib

def source(X):
    """
    Compute the value of a 2D Gaussian function at given points, centered at the Mountbatten Building.

    Parameters:
    - X: array-like, coordinates.
    

    Returns:
    - Gaussian values at the specified (x, y) coordinates.
    """
    x0=442365.
    y0=115483.
    sigma_x=500.
    sigma_y=500.
    amplitude=1.
    return amplitude * np.exp(-(((X[0] - x0)**2) / (2 * sigma_x**2) + ((X[1] - y0)**2) / (2 * sigma_y**2)))

def staticdiffusionsolver(fn,fi,fb,res,plot=True):
    plotcheck=plot

    nodes = np.loadtxt(fn)
    IEN = np.loadtxt(fi, dtype=np.int64)
    boundary_nodes = np.loadtxt(fb, dtype=np.int64)
    
    #setting homogeneous Dirichlet BCs on boundary nodes whose y<y_soton
    lower_bdry=[0]
    for i in boundary_nodes:
        if (i!=0) and (nodes[i,1]<110000) and ((nodes[i,1]-115483)<=50):
            lower_bdry=np.append(lower_bdry,boundary_nodes[i])
        
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
        k_e = lib.local_stiffness(nodes[IEN[e,:],:])
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
    Psi_interior = np.linalg.solve(K, F)
    Psi_A = np.zeros(N_nodes)
    for n in range(N_nodes):
        if ID[n] >= 0: # Otherwise Psi should be zero, and we've initialized that already.
            Psi_A[n] = Psi_interior[ID[n]]
    
    #map plot
    if (plotcheck):
    
        plt.cla()
        plt.tripcolor(nodes[:,0], nodes[:,1], Psi_A, triangles=IEN)
        
        plt.scatter([442365,473993],[115483,171625], s=5, c='red')
        plt.scatter(nodes[lower_bdry,0],nodes[lower_bdry,1], s=2)
        #plt.scatter(nodes[:,0],nodes[:,1], s=1)
        
        plt.xlabel(r"longitude", fontsize=12)
        plt.ylabel(r"latitude", fontsize=12)
        
        plt.tick_params(labelleft=False,labelbottom=False)
        plt.colorbar()
        plt.show()
    
    #computation of the target quantities
    psi_R=lib.psi_r(nodes, IEN, Psi_A)
    psi_S=lib.psi_s(nodes, IEN, Psi_A)
    
    Psi_R=psi_R/psi_S
    sqrtNe=np.sqrt(N_nodes)
    #sqrtDOF=np.sqrt(N_equations)
    
    return N_equations, Psi_R

fileN='las_grids/las_nodes_1_25k.txt'
fileI='las_grids/las_ien_1_25k.txt'
fileB='las_grids/las_bdry_1_25k.txt'
res=1.25

DOF1_25, Psi1_25 = staticdiffusionsolver(fileN, fileI, fileB, res)
print("The grid considered for this computation has average element side length equal to",res,"km.")
print("The pollutant concentration over Reading, normalized to one over Southampton, is %.3f."%Psi1_25)

#%% 
def accuracy(maxDOF,bestPsi):
    
    files_nodes={
           "2_5km": 'las_grids/las_nodes_2_5k.txt',
           "5km": 'las_grids/las_nodes_5k.txt',
           "10km": 'las_grids/las_nodes_10k.txt',
           "20km": 'las_grids/las_nodes_20k.txt',
           "40km": 'las_grids/las_nodes_40k.txt'
           }
    
    files_ien={
           "2_5km": 'las_grids/las_ien_2_5k.txt',
           "5km": 'las_grids/las_ien_5k.txt',
           "10km": 'las_grids/las_ien_10k.txt',
           "20km": 'las_grids/las_ien_20k.txt',
           "40km": 'las_grids/las_ien_40k.txt'
           }

    files_bdry={
           "2_5km": 'las_grids/las_bdry_2_5k.txt',
           "5km": 'las_grids/las_bdry_5k.txt',
           "10km": 'las_grids/las_bdry_10k.txt',
           "20km": 'las_grids/las_bdry_20k.txt',
           "40km": 'las_grids/las_bdry_40k.txt'
           }

    resolutions={
           "2_5km": 2.5,
           "5km": 5,
           "10km": 10,
           "20km": 20,
           "40km": 40
           }

    Psi_R=[bestPsi]
    DOF=[maxDOF]

    for file in files_nodes:
        
        fn=files_nodes[file]
        fi=files_ien[file]
        fb=files_bdry[file]
        res=resolutions[file]
        
        dof, Psi = staticdiffusionsolver(fn, fi, fb, res, plot=False)
        
        Psi_R.append(Psi)
        DOF.append(dof)
        
    y_4N=Psi_R[0]
    y_2N=Psi_R[1]
    y_N=Psi_R[2]
    
    err_wrt_best=np.zeros(6)
    
    for i in range(len(err_wrt_best)):
        err_wrt_best[i]=abs(Psi_R[i]-Psi_R[0])
        
    #print(np.log(DOF))
    #print(np.log(err_wrt_best))    
    polyfit_coeffs = np.polyfit(np.log(DOF[1:]),np.log(err_wrt_best[1:]),1) 

    trendline = lambda data,x: np.poly1d(data)(x)
    
    plt.figure()
    plt.loglog(DOF, err_wrt_best, 'xk')
    plt.loglog(DOF, np.exp(trendline(polyfit_coeffs,np.log(DOF))),'-r', 
               label=rf'$\propto N^{{{polyfit_coeffs[0]:.2f}}}$')
    plt.xlabel('Degrees of freedom')
    plt.ylabel('Error relative to max res soln')
    plt.legend()
    plt.grid()
    plt.title(f'Relative Error in static case')
    
    #x = np.array([40000,20000,10000,5000,2500,1250])
    #polyfit_coeffs = np.polyfit(np.log(x[1:]),np.log(err_wrt_best[1:]),1) 
    
    #plt.figure()
    #plt.loglog(x, err_wrt_best, 'xk')
    #plt.loglog(x, np.exp(trendline(polyfit_coeffs,np.log(x))),'-r', 
    #           label=rf'$\propto x^{{{polyfit_coeffs[0]:.2f}}}$')
    #plt.xlabel('Degrees of freedom')
    #plt.ylabel('Error relative to max res soln')
    #plt.legend()
    #plt.grid()
    #plt.title(f'Relative Error in static case')
    
    plt.show()   

accuracy(DOF1_25,Psi1_25)       