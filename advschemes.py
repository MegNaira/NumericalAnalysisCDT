#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:34:36 2024

@author: arianna
"""
import numpy as np
import matplotlib.pyplot as plt


class whiz:
    
    def __init__(self, u=1.0, nx=50, nt=250, x=None):
        #Parameters
        self.u = u
        self.nx = nx
        self.nt = nt 
        
        #Other derived parameters
        self.dx = 1./self.nx
        self.dt = 1./self.nt
        self.c = self.dt*self.u/self.dx
        
        self.x = np.linspace(0.0, 1.0, self.nx+1)
        
        
        self.phi=self.analytic(0)
        self.phiOld=self.phi.copy()
        self.phiOlder=self.phi.copy()
        
    #the analytical solution to the advection equation, used for comparison and for setting initial conditions
    def analytic(self,t):
        return np.where((self.x-self.u*t)%1. < 0.5, np.power(np.sin(2*np.pi*(self.x-self.u*t)),2), 0.)
    
    
    #function to plot interesting time steps
    def plot_timestep(self,n,scheme):
        fig, ax = plt.subplots(1,1,figsize=(14,6))
        ax.cla()
        ax.plot(self.x,self.analytic((n+1)*self.dt),color='#E98A15',label='Analytical')
        ax.plot(self.x, self.phi, color='#59114D', label = 'Numerical')
        ax.legend(loc = 'upper right', fontsize=14)
        ax.text(0.88,0.78,'t=%.2f'%((n+1)*self.dt), fontsize=16)
        ax.set_xlabel('x', fontsize=20)
        ax.tick_params(axis='x', which='major', labelsize=18, width=3, length=7)
        ax.tick_params(axis='x', which='minor', labelsize=0, width=2, length=5)
        ax.tick_params(axis='y', which='major', labelsize=18, width=3, length=7)
        ax.tick_params(axis='y', which='minor', labelsize=0, width=2, length=3)
        ax.set_ylabel('$\phi$', fontsize=20)
        ax.set_title(scheme + ', c=%.1f'%(self.c), fontsize=22)
        ax.set_ylim([0,1])
        fig.savefig(scheme + "_step%i.jpg"%n)
        plt.show()
        plt.pause(0.05)
    
    #FTBS - computes the numerical solution and calls the plotting function
    def ftbs(self):
        scheme='FTBS'
        
        for n in range(self.nt):
            for j in range(1,self.nx+1):
                self.phi[j] = self.phiOld[j] - self.c*(self.phiOld[j] - self.phiOld[j-1])
            self.phi[0]=self.phi[-1] 
            self.phiOld = self.phi.copy()
            
            #selecting and plotting 10 timesteps
            y=self.nt/10
            if n%y==0:
                self.plot_timestep(n,scheme)
                
    #FTCS - computes the numerical solution and calls the plotting function
    def ftcs(self):
        scheme='FTCS'
        
        for n in range(self.nt):
            for j in range(1,self.nx):
                self.phi[j] = self.phiOld[j] - self.c*(self.phiOld[j+1] - self.phiOld[j-1])/2
            self.phi[0]=self.phiOld[0]-self.c*(self.phiOld[1]-self.phiOld[-2])/2    
            self.phi[-1]=self.phi[0] 
            self.phiOld = self.phi.copy()
            
            #selecting and plotting 10 timesteps
            y=self.nt/10
            if n%y==0:
                self.plot_timestep(n,scheme)
    
    #CTCS - computes the numerical solution and calls the plotting function
    def ctcs(self):
        scheme='FTCS'
        self.phiOld=self.analytic(self.dt)
        
        for n in range(2,self.nt):
            for j in range(1,self.nx):
                self.phi[j] = self.phiOlder[j] - self.c*(self.phiOld[j+1] - self.phiOld[j-1])
            self.phi[0]=self.phiOlder[0]-self.c*(self.phiOld[1]-self.phiOld[-2])   
            self.phi[-1]=self.phi[0] 
            self.phiOlder = self.phiOld.copy()
            self.phiOld = self.phi.copy()
            
            #selecting and plotting 10 timesteps
            y=self.nt/10
            if n%y==0:
                self.plot_timestep(n,scheme) 
        