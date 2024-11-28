#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:48:05 2024

@author: arianna
"""

import matplotlib.pyplot as plt
import numpy as np
import librarystatic2d as lib

#GRID GENERATOR
def generate_2d_grid(Nx):
    
    Nnodes = Nx+1
    x = np.linspace(0, 1, Nnodes)
    y = np.linspace(0, 1, Nnodes)
    X, Y = np.meshgrid(x,y)
    nodes = np.zeros((Nnodes**2,2))
    nodes[:,0] = X.ravel()
    nodes[:,1] = Y.ravel()
    ID = np.zeros(len(nodes), dtype=np.int64)
    boundaries = dict()  # Will hold the boundary values
    n_eq = 0
    for nID in range(len(nodes)):
        if np.allclose(nodes[nID, 0], 0):
            ID[nID] = -1
            boundaries[nID] = 0  # Dirichlet BC
        else:
            ID[nID] = n_eq
            n_eq += 1
            if ( (np.allclose(nodes[nID, 1], 0)) or 
                 (np.allclose(nodes[nID, 0], 1)) or 
                 (np.allclose(nodes[nID, 1], 1)) ):
                boundaries[nID] = 1 # Neumann BC
                pass
    IEN = np.zeros((2*Nx**2, 3), dtype=np.int64)
    for i in range(Nx):
        for j in range(Nx):
            IEN[2*i+2*j*Nx  , :] = (i+j*Nnodes, 
                                    i+1+j*Nnodes, 
                                    i+(j+1)*Nnodes)
            IEN[2*i+1+2*j*Nx, :] = (i+1+j*Nnodes, 
                                    i+1+(j+1)*Nnodes, 
                                    i+(j+1)*Nnodes)
    return nodes, IEN, ID, boundaries


nodes,IEN,ID,boundary_nodes = generate_2d_grid(10)

#DIFFERENT POSSIBLE SOURCES AND THEIR RESPECTIVE ANALYTICAL SOLUTIONS
def S1(X):
    X = np.atleast_2d(X)
    return 1.0

def exact1(X):
    X = np.atleast_2d(X)
    x = X[:,0]
    y = X[:,1]
    return np.array(x*(1-x/2))

def S2(X):
    X = np.atleast_2d(X)
    x = X[:,0]
    y = X[:,1]
    return np.array(2*x*(x-2)*(3*y**2-3*y+1/2)+y**2*(y-1)**2)

def exact2(X):
    X = np.atleast_2d(X)
    x = X[:,0]
    y = X[:,1]
    return np.array(x*(1-x/2)*y**2*(1-y)**2) 

# n_eq = 0
# for i in range(len(nodes[:, 1])):
#     if i in boundary_nodes:
#         ID[i] = -1
#     else:
#         ID[i] = n_eq
#         n_eq += 1

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
    f_e = lib.local_force(S1, nodes[IEN[e,:],:])
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
        
fig, ax=plt.subplots(1,2, figsize=(10,5))


ax[0].tripcolor(nodes[:,0], nodes[:,1], Psi_A, triangles=IEN)
ax[0].set_xlabel(r"x", fontsize=15)
ax[0].set_ylabel(r"y", fontsize=15)
ax[0].set_title(r"Numerical")


ax[1].tripcolor(nodes[:,0], nodes[:,1], exact1(nodes), triangles=IEN)
ax[1].set_xlabel(r"x", fontsize=15)
ax[1].set_title(r"Analytical")


fig.tight_layout()  
