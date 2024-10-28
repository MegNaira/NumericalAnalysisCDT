#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:19:11 2024

@author: arianna
"""

import numpy as np
import matplotlib.pyplot as plt
from advschemes import analytic, ftbs, ftcs, ctcs

def main():
    "Solve the advection equation"
    #Parameters
    u = 1.0
    nx = 50
    nt = 250
    
    #Other derived parameters
    dx = 1./nx # The spacial resolution
    dt = 1./nt # The time step
    c = u*dt/dx
    print('Courant number =', c)

    #space points
    x = np.linspace(0.0, 1.0, nx + 1)
    
    #calling the schemes
    phi_FTBS = ftbs(u,x,nx,nt,dx,dt)
    phi_FTCS = ftcs(u,x,nx,nt,dx,dt)
    phi_CTCS = ctcs(u,x,nx,nt,dx,dt)
    
    
    "Check on stability."
    "FTBS is stable and damping for 0<=c=1. It is stable and not damping for c=1. It is first order accurate."
    "CTCS is stable and not damping for 0<=c<=1. It is second order accurate."
    "FTCS is unconditionally unstable."


main()
                
