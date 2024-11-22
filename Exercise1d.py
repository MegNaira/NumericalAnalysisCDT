#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:19:44 2024

@author: arianna
"""

import numpy as np
import matplotlib.pyplot as plt


"""
Any node that is not to be included (as its value is given by a Dirichlet boundary condition) has its associated equation number A set to -1.
The first node that must be included is given value 0. 
We then go element-by-element: the left-hand node of element e is the same as the right-hand node of element e-1, 
so picks up the same equation number. 
The right-hand node of element e, if considered, then has equation number one higher than the left-hand node of that element.
"""

#1. Set the number of elements
Ne=20

# Define analytic solution for test cases
def analytic1(x):
    return x*(x**2-3*x+3)/6

def analytic2(x):
    return x*(4-6*x+4*x**2-x**3)/12

# Define source function S(x) for test cases
def source1(x):
    return 1 - x

def source2(x):
    return (1 - x)**2

def source3(x):
    return 1 if abs(x - 0.5) < 0.25 else 0


#2. Define the domain and set node locations
def nodes():
    return np.linspace(0,1,Ne+1)

def ref_coord():
    return np.linspace(-1,1,2)    
    
def pick_element(e):
    return nodes()[e:e+2]

def location():
    # first index is either 0 for left or 1 for right
    LM = np.zeros((2, Ne), dtype=np.int64)
    for e in range(Ne):
        if e==0:
            # Treat first element differently due to BC
            LM[0,e] = -1 # Left hand node of first element is not considered thanks to BC.
            LM[1,e] = 0 #the first equation
        else:
            # Left node of this element is right node of previous element
            LM[0,e] = LM[1,e-1]
            LM[1,e] = LM[0,e] + 1
    
    return LM

def stiffness(e):
    x_e = pick_element(e)
    print(e,x_e)
    return np.array([[1,-1],
                     [-1,1]])/(x_e[1]-x_e[0])

def force(e):
    x_e = pick_element(e)
    S1 = source2(x_e[1])
    S2 = source2(x_e[0])
    return np.array([2*S1+S2,S1+2*S2])*(x_e[1]-x_e[0])/6
              

def solver():
    #boundary conditions
    alpha=0
    beta=0
    #3. Set up the location matrix
    LM = location()
    #5. Set up arrays, initally all zero
    psi = np.zeros_like(nodes())
    K = np.zeros((Ne,Ne))
    F = np.zeros(Ne)
    
    #6. For each element:
    for e in range(Ne):
        #6.1 Form the element stiffness matrix
        k_e = stiffness(e)
        #6.2 Form the element force vector
        f_e = force(e)
        #6.3 Add the contributions to global stiffness matrix and force vector
        for a in range(2):
            A = LM[a,e]
            if (A>=0):
                F[A]+=f_e[a]
            for b in range(2):
                B = LM[b,e]
                if (A>=0) and (B>=0):
                    K[A,B]+=k_e[a,b]
        #6.4 Modify force vector for Dirichelet BC        
        if e==0:
            F[0]-=alpha*k_e[1,0]
    #6.4 Modify force vector for Neumann BC
    F[-1]+=beta
    
    #7. Solve
    psi[0]=alpha
    psi[1:]=np.linalg.solve(K,F)
    
    return psi
    
def plot():
    x=np.linspace(0,1,100)
    plt.plot(x, analytic2(x),label='analytical', c='#390099')
    #plt.plot(nodes(),solver(),label='numerical', c='#FF5400')
    plt.scatter(nodes(),solver(),label='numerical',s=20,c='#FF5400')
    plt.grid()
    plt.legend()
    plt.show()

plot()
    
    
    