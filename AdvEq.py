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
    Solves the advection equation calling three finite difference numerical methods
    
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


def checks(err=False,stb=False,acc=False):
    
    """
    Each flag specifies the check we wish to do.
    They are all False by default.
    
    If err=True,
    then the function computes and plots the evolution with time 
    of the L2 norm between analytical and numerical solutions, 
    for fixed Courant number.
    
    If stb=True,
    then the function computes and plots the dependence on the Courant number 
    of the L2 norm between analytical and numerical solutions,
    for fixed time = 1.0 (endtime).
    
    If acc=True,
    then the function computes the dependence on dx
    of the L2 norm between analytical and numerical solutions,
    for fixed Courant number.
    
    
    When the Courant number is fixed, its value is the default value for the class, i.e. 0.4.
    It can be changed by specifying different values for nx and nt when the class is called,
    keeping in mind that in this experiment c=nx/nt.
    """
    
    fig, ax = plt.subplots(1,1,figsize=(12,10))
    
    
    #Disclaimer: I call the class extensively when using its functions to avoid the class' "mutability",
    #i.e. changes to it that affect the same memory location, when calling at once multiple functions from it.
    #It's a bug that, unfortunetely, has presented to me, and that I haven't been able to solve in other ways for the moment.
    
    if (err):
        
        #initializing the class to calculate errors at every timestep
        w1 = advschemes.whiz(plot=False,errors=True)
        
        t_ftbs, err_ftbs = advschemes.whiz(plot=False,errors=True).ftbs()
        t_ftcs, err_ftcs = advschemes.whiz(plot=False,errors=True).ftcs()
        t_ctcs, err_ctcs = advschemes.whiz(plot=False,errors=True).ctcs()
        
        #plot specifics
        ax.cla()
        ax.plot(t_ftbs, err_ftbs, color='#B80053', lw=3, label='FTBS')
        ax.plot(t_ftcs, err_ftcs, color='#EAC435', lw=3, label='FTCS')
        ax.plot(t_ctcs, err_ctcs, color='#345995', lw=3, label='CTCS')
        ax.legend(loc = 'upper left', fontsize=20)
        ax.set_xlabel('time', fontsize=20)
        ax.set_ylabel('$L_2$-norm of errors', fontsize=20)
        ax.text(0.2,0.5,'c=%.1f'%(w1.c), fontsize=18)
        
        ax.tick_params(axis='x', which='major', labelsize=18, width=3, length=7)
        ax.tick_params(axis='x', which='minor', labelsize=0, width=2, length=5)
        ax.tick_params(axis='y', which='major', labelsize=18, width=3, length=7)
        ax.tick_params(axis='y', which='minor', labelsize=0, width=2, length=3)
        
        plt.grid()
        
        fig.savefig("ErrorEvolution_c%.1f.jpg"%w1.c)
        plt.show
        
        
        
    if (stb):
        
        #grid of values for the Courant number
        c=np.linspace(0.1,1.4,100)
        
        l2norm_ftbs=np.zeros(len(c))
        l2norm_ftcs=np.zeros(len(c))
        l2norm_ctcs=np.zeros(len(c))
        
        #loop over the values for c
        for j in range(len(c)):
            
            w2=advschemes.whiz(nx=60, nt=60/c[j], plot=False)
            
            exp = advschemes.whiz(nx=60, nt=60/c[j], plot=False).analytic(w2.nt*w2.dt)
        
            phi_ftbs = advschemes.whiz(nx=60, nt=60/c[j], plot=False).ftbs()
            phi_ftcs = advschemes.whiz(nx=60, nt=60/c[j], plot=False).ftcs()
            phi_ctcs = advschemes.whiz(nx=60, nt=60/c[j], plot=False).ctcs()
            
            l2norm_ftbs[j]=advschemes.whiz(nx=60, nt=60/c[j], plot=False).ltwo(phi_ftbs,exp,1/w2.nx)
            l2norm_ftcs[j]=advschemes.whiz(nx=60, nt=60/c[j], plot=False).ltwo(phi_ftcs,exp,1/w2.nx)
            l2norm_ctcs[j]=advschemes.whiz(nx=60, nt=60/c[j], plot=False).ltwo(phi_ctcs,exp,1/w2.nx)
       
        #plot specifics    
        ax.cla()
        ax.plot(c,l2norm_ftbs, color='#B80053', lw=3, label='FTBS')
        ax.plot(c,l2norm_ftcs, color='#EAC435', lw=3, label='FTCS')
        ax.plot(c,l2norm_ctcs, color='#345995', lw=3, label='CTCS')
        ax.legend(loc = 'upper left', fontsize=20)
        ax.set_xlabel('Courant number', fontsize=20)
        ax.set_ylabel('$L_2$-norm of errors', fontsize=20)
        ax.text(0.1,2.3,'$t=t_{end}$=%.1f'%(w2.nt*w2.dt), fontsize=18)
        
        ax.tick_params(axis='x', which='major', labelsize=18, width=3, length=7)
        ax.tick_params(axis='x', which='minor', labelsize=0, width=2, length=5)
        ax.tick_params(axis='y', which='major', labelsize=18, width=3, length=7)
        ax.tick_params(axis='y', which='minor', labelsize=0, width=2, length=3)
        
        
        fig.savefig("SchemesStability_tend%.1f.jpg"%(w2.dt*w2.nt))
        plt.grid()
        plt.show
        
        
        
    if (acc):
        
        nx=np.array([30,40,50,60,70,80])
        nt=nx/0.4 #nt=nx/c
        
        diff_ftbs=np.zeros(len(nx))
        diff_ctcs=np.zeros(len(nx)) 

        #fitting function
        def retta(x,m,q):
            return m*x+q
        
        for i in range(len(nx)):
            
            w3=advschemes.whiz(nx=nx[i], nt=nt[i], plot=False)

            res_ftbs=advschemes.whiz(nx=nx[i], nt=nt[i], plot=False).ftbs()
            res_ctcs=advschemes.whiz(nx=nx[i], nt=nt[i], plot=False).ctcs()
            
            exp=advschemes.whiz(nx=nx[i], nt=nt[i], plot=False).analytic(w3.nt*w3.dt)
        
            #diff_ftbs[i]=w3.rmse(res_ftbs,exp)
            #diff_ctcs[i]=w3.rmse(res_ctcs,exp)
            
            diff_ftbs[i]=advschemes.whiz(nx=nx[i], nt=nt[i], plot=False).ltwo(res_ftbs,exp,1/nx[i])
            diff_ctcs[i]=advschemes.whiz(nx=nx[i], nt=nt[i], plot=False).ltwo(res_ctcs,exp,1/nx[i])
            
        #FIT to find the order of convergence given by the slope on a loglog plane.
        
        #[m_ftbs,q_ftbs]= np.polyfit(np.log10(1/nx), np.log10(diff_ftbs),1)
        #[m_ctcs,q_ctcs]= np.polyfit(np.log10(1/nx), np.log10(diff_ctcs),1)
        
        [m_ftbs,q_ftbs], pcov_ftbs = opt.curve_fit(retta,np.log10(1/nx), np.log10(diff_ftbs))
        [m_ctcs,q_ctcs], pcov_ctcs = opt.curve_fit(retta,np.log10(1/nx), np.log10(diff_ctcs))
        
        
        #plot specifics
        ax.cla()
        ax.scatter(np.log10(1/nx), np.log10(diff_ftbs), color='#B80053', s=100, label='FTBS')
        ax.scatter(np.log10(1/nx), np.log10(diff_ctcs), color='#345995', s=100, label = 'CTCS')
        ax.plot(np.log10(1/nx),m_ftbs*np.log10(1/nx)+q_ftbs, color='#B80053')
        ax.plot(np.log10(1/nx),m_ctcs*np.log10(1/nx)+q_ctcs, color='#345995')
        ax.legend(loc = 'lower right', fontsize=20)
        ax.text(-1.55,-0.4,'n=%.2f'%m_ftbs, color='#B80053', fontsize=16, rotation=10)
        ax.text(-1.55,-0.95,'n=%.2f'%m_ctcs, color='#345995', fontsize=16, rotation=30)
        ax.text(-1.6,-1.6,'c=%.1f'%(w3.c), fontsize=18)
        ax.set_xlabel('log(dx)', fontsize=20)
        ax.set_ylabel('log($L_2$-norm of errors)', fontsize=20)
        
        ax.tick_params(axis='x', which='major', labelsize=18, width=3, length=7)
        ax.tick_params(axis='x', which='minor', labelsize=0, width=2, length=5)
        ax.tick_params(axis='y', which='major', labelsize=18, width=3, length=7)
        ax.tick_params(axis='y', which='minor', labelsize=0, width=2, length=3)
        
        plt.grid()
            
        fig.savefig("Convergence_c%.1f.jpg"%w3.c)
        plt.show()
        
    
#main()
checks(acc=True)
                
