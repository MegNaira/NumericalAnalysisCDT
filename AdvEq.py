#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:19:11 2024

@author: arianna
"""

import numpy as np
import matplotlib.pyplot as plt
import advschemes

def main():
    """"
    Solves the advection equation using three finite difference numerical methods
    """

    w=advschemes.whiz()
    
    schemes = {
        "FTBS" : w.ftbs,
        "FTCS" : w.ftcs,
        "CTCS" : w.ctcs
        }
    
    #calling the schemes
    for method in schemes:
        result=schemes[method]()


def main_sw():
    """
    Same as main, but with a squarewave as initial condition, so to see the effect of discontinuities

    """
    w=advschemes.whiz(squarewave=True)
    
    schemes = {
        "FTBS" : w.ftbs,
        "FTCS" : w.ftcs,
        "CTCS" : w.ctcs
        }
    
    #calling the schemes
    for method in schemes:
        result=schemes[method]() 
    
    
    "Check on stability. How to present it?"
    "FTBS is stable and damping for 0<=c=1. It is stable and not damping for c=1. It is first order accurate."
    "CTCS is stable and not damping for 0<=c<=1. It is second order accurate."
    "FTCS is unconditionally unstable."

def accuracy():
    s=[1.0,2.0] #when it's equal to 2, it means that I double the resolution and halve the timestep while keeping nt constant
    nx=[50,150]
    nt=[250,750]
    
    err_ftbs=[]
    err_ctcs=[]
    dx=[]
    

    
    for i in range(len(s)):
        y=advschemes.whiz(nx=nx[0], nt=nt[0], s=s[i], plot=False)
        
        dx.append(y.dx)
    
        res_ftbs=y.ftbs()
        res_ctcs=y.ctcs()
            
        err_ftbs.append(y.rmse(res_ftbs,y.analytic(y.nt*y.dt)))
        err_ctcs.append(y.rmse(res_ctcs,y.analytic(y.nt*y.dt)))

    fig, ax = plt.subplots(1,1,figsize=(12,10))
    ax.cla()
    ax.scatter(dx, err_ftbs, color='#802392', s=100, label='FTBS')
    ax.scatter(dx, err_ctcs, color='#A5F8D3', s=100, label = 'CTCS')
    ax.legend(loc = 'upper right', fontsize=14)
    #ax.text(0.88,0.78,'t=%.2f'%((n+1)*self.dt), fontsize=16)
    ax.set_xlabel('dx', fontsize=20)
    ax.set_ylabel('RMSE', fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.tick_params(axis='x', which='major', labelsize=18, width=3, length=7)
    ax.tick_params(axis='x', which='minor', labelsize=0, width=2, length=5)
    ax.tick_params(axis='y', which='major', labelsize=18, width=3, length=7)
    ax.tick_params(axis='y', which='minor', labelsize=0, width=2, length=3)

    ax.set_title('Order of convergence, nt=%i'%(y.nt), fontsize=22)
    #ax.set_ylim([-0.1,1.1])
    
    plt.grid()
        
    fig.savefig("Convergence_nt%i.jpg"%y.nt)
    plt.show()
    
        
    
main()
accuracy()
                
