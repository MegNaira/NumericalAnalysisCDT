#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:19:11 2024

@author: arianna
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import advschemes


def main():
    """"
    Solves the advection equation using three finite difference numerical methods
    """

    w=advschemes.whiz()
    
    schemes = {
        "FTBS" : w.ftbs,
        #"FTCS" : w.ftcs,
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

def retta(x,m,q):
    return m*x+q

def accuracy():
    
    nx=np.array([30,40,50,60,70,80])
    nt=nx/0.4
    
    err_ftbs=np.zeros(len(nx))
    err_ctcs=np.zeros(len(nx))   
    
    for i in range(len(nx)):
        
        y=advschemes.whiz(nx=nx[i], nt=nt[i], plot=False)

        res_ftbs=advschemes.whiz(nx=nx[i], nt=nt[i], plot=False).ftbs()
        res_ctcs=advschemes.whiz(nx=nx[i], nt=nt[i], plot=False).ctcs()
        
        exp=y.analytic(y.nt*y.dt)
    
        #err_ftbs[i]=y.rmse(res_ftbs,exp)
        #err_ctcs[i]=y.rmse(res_ctcs,exp)
        
        err_ftbs[i]=y.ltwo(res_ftbs,exp,1/nx[i])
        err_ctcs[i]=y.ltwo(res_ctcs,exp,1/nx[i])
            
        #err_ftbs[i]=np.sqrt(np.sum((res_ftbs-exp)**2/nx[i])/np.sum((exp/nx[i])**2))
        #err_ctcs[i]=np.sqrt(np.sum((res_ctcs-exp)**2/nx[i])/np.sum((exp/nx[i])**2))
    
    #[m_ftbs,q_ftbs]= np.polyfit(np.log10(1/nx), np.log10(err_ftbs),1)
    #[m_ctcs,q_ctcs]= np.polyfit(np.log10(1/nx), np.log10(err_ctcs),1)
    
    [m_ftbs,q_ftbs], pcov_ftbs = opt.curve_fit(retta,np.log10(1/nx), np.log10(err_ftbs))
    [m_ctcs,q_ctcs], pcov_ctcs = opt.curve_fit(retta,np.log10(1/nx), np.log10(err_ctcs))
        
    print(m_ftbs,m_ctcs)
    
    fig, ax = plt.subplots(1,1,figsize=(12,10))
    ax.cla()
    ax.scatter(np.log10(1/nx), np.log10(err_ftbs), color='#B80053', s=100, label='FTBS')
    ax.scatter(np.log10(1/nx), np.log10(err_ctcs), color='#345995', s=100, label = 'CTCS')
    ax.plot(np.log10(1/nx),m_ftbs*np.log10(1/nx)+q_ftbs, color='#B80053')
    ax.plot(np.log10(1/nx),m_ctcs*np.log10(1/nx)+q_ctcs, color='#345995')
    ax.legend(loc = 'lower right', fontsize=20)
    ax.text(-1.55,-1.4,'n=%.2f'%m_ftbs, color='#B80053', fontsize=16, rotation=10)
    ax.text(-1.55,-1.95,'n=%.2f'%m_ctcs, color='#345995', fontsize=16, rotation=30)
    ax.set_xlabel('log(dx)', fontsize=20)
    ax.set_ylabel('log(L2 norm)', fontsize=20)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    
    ax.tick_params(axis='x', which='major', labelsize=18, width=3, length=7)
    ax.tick_params(axis='x', which='minor', labelsize=0, width=2, length=5)
    ax.tick_params(axis='y', which='major', labelsize=18, width=3, length=7)
    ax.tick_params(axis='y', which='minor', labelsize=0, width=2, length=3)

    ax.set_title('Order of convergence, c=%.1f'%(y.c), fontsize=22)
    #ax.set_ylim([-0.1,1.1])
    
    plt.grid()
        
    fig.savefig("ConvergenceL2_c%.1f.jpg"%y.c)
    plt.show()
    
        
    
#main()
accuracy()
                
