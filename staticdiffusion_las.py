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
    sigma_x=50.
    sigma_y=50.
    amplitude=1.
    return amplitude * np.exp(-(((X[0] - x0)**2) / (2 * sigma_x**2) + ((X[1] - y0)**2) / (2 * sigma_y**2)))


nodes = np.loadtxt('las_grids/las_nodes_5k.txt')
IEN = np.loadtxt('las_grids/las_IEN_5k.txt', dtype=np.int64)
boundary_nodes = np.loadtxt('las_grids/las_bdry_5k.txt', dtype=np.int64)

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
        

plt.tripcolor(nodes[:,0], nodes[:,1], Psi_A, triangles=IEN)
plt.scatter([442365,473993],[115483,171625], s=5, c='red')
plt.scatter(nodes[lower_bdry,0],nodes[lower_bdry,1], s=2)
plt.xlabel(r"longitude", fontsize=12)
plt.ylabel(r"latitude", fontsize=12)
#plt.title(r"Numerical")
plt.tick_params(labelleft=False,labelbottom=False)
plt.colorbar()

 
