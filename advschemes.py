#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:34:36 2024

@author: arianna
"""
import numpy as np
import matplotlib.pyplot as plt

#the analytical solution to the advection equation, used for comparison and for setting initial conditions
def analytic(u,x,t):
    return np.where((x-u*t)%1. < 0.5, np.power(np.sin(2*np.pi*(x-u*t)),2), 0.)

#function to plot interesting time steps
def plot_timestep(x,dx,dt,n,phi,u,scheme):
    fig, ax = plt.subplots(1,1,figsize=(14,6))
    ax.cla()
    ax.plot(x,analytic(u,x,(n+1)*dt),color='#E98A15',label='Analytical')
    ax.plot(x, phi, color='#59114D', label = 'Numerical')
    ax.legend(loc = 'upper right', fontsize=14)
    ax.text(0.88,0.78,'t=%.2f'%((n+1)*dt), fontsize=16)
    ax.set_xlabel('x', fontsize=20)
    ax.tick_params(axis='x', which='major', labelsize=18, width=3, length=7)
    ax.tick_params(axis='x', which='minor', labelsize=0, width=2, length=5)
    ax.tick_params(axis='y', which='major', labelsize=18, width=3, length=7)
    ax.tick_params(axis='y', which='minor', labelsize=0, width=2, length=3)
    ax.set_ylabel('$\phi$', fontsize=20)
    ax.set_title(scheme + ', c=%.1f'%(u*dt/dx), fontsize=22)
    ax.set_ylim([0,1])
    fig.savefig(scheme + "_step%i.jpg"%n)
    plt.show()
    plt.pause(0.05)

#FTBS - sets the initial condition, computes the numerical solution and calls the plotting function
def ftbs(u,x,nx,nt,dx,dt):
    scheme='FTBS'
    #initial condition
    phiInit = np.array(analytic(u,x,0))
    
    phiOld = phiInit.copy()
    phi=phiOld.copy()
    c = u*dt/dx
    
    for n in range(nt):
        for j in range(1,nx+1):
            phi[j] = phiOld[j] - c*(phiOld[j] - phiOld[j-1])
        phi[0]=phi[-1] 
        phiOld = phi.copy()
        
        #selecting and plotting timesteps
        y=nt/10
        if n%y==0:
            plot_timestep(x,dx,dt,n,phi,u,scheme)
    return phi       

#FTCS
def ftcs(u,x,nx,nt,dx,dt):
    scheme='FTCS'
    #initial condition
    phiInit = np.array(analytic(u,x,0))
    
    phiOld = phiInit.copy()
    phi=phiOld.copy()
    c = u*dt/dx
    
    for n in range(nt):
        for j in range(1,nx):
            phi[j] = phiOld[j] - c*(phiOld[j+1] - phiOld[j-1])/2
        phi[0]=phiOld[0]-c*(phiOld[1]-phiOld[-2])/2    
        phi[-1]=phi[0] 
        phiOld = phi.copy()
        
        #selecting and plotting timesteps
        y=nt/10
        if n%y==0:
            plot_timestep(x,dx,dt,n,phi,u,scheme)
    return phi  


#CTCS
def ctcs(u,x,nx,nt,dx,dt):
    scheme='CTCS'
    #initial condition
    phiInit = np.array(analytic(u,x,0))
    phiInit1 = np.array(analytic(u,x,dt))
    
    phiOlder = phiInit.copy()
    phiOld = phiInit1.copy()
    phi=phiOlder.copy()
    c = u*dt/dx
    
    for n in range(2,nt):
        for j in range(1,nx):
            phi[j] = phiOlder[j] - c*(phiOld[j+1] - phiOld[j-1])
        phi[0]=phiOlder[0]-c*(phiOld[1]-phiOld[-2])   
        phi[-1]=phi[0] 
        phiOlder = phiOld.copy()
        phiOld = phi.copy()
        
        #selecting and plotting timesteps
        y=nt/10
        if n%y==0:
            plot_timestep(x,dx,dt,n,phi,u,scheme)
    return phi  
            
        