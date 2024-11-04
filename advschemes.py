#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:34:36 2024

@author: arianna
"""
import numpy as np
import matplotlib.pyplot as plt


class whiz:
    
    """
    This class is used to implement different schemes to solve
    the same partial differential equation, namely the advection equation.
    
    It takes 5 optional arguments:
        
        floats:
        u, the wind speed (default=1.0);
        
        int:
        nx, the number or spatial points used to discretise the wave (default=50);
        nt, the number of time steps (default=125)
        
        bool:
        plot, activates the timestep plotting function when True (default=True);
        errors, if True, the schemes give as output the errors at every time step,
                if False, the output is the numerical solution at the last time step (default=False).
                
    It contains 7 callable functions:
        
        analytic, computes the analytic solution to the advection equation;
        plot_timestep, used by the schemes to plot the results at different time steps;
        ftbs, computes the numerical solution using the forward in time backwards in space finite differences scheme;
        ftcs, computes the numerical solution using the forward in time centered in space finite differences scheme;
        ctcs, computes the numerical solution using the centered in time centered in space finite differences scheme;
        ltwo, computes the L2 norm of the difference between the analytical and the numerical solution.
    
    """
    
    def __init__(self, u=1.0, nx=50, nt=50/0.4, plot=True, errors=False):
        
        #Parameters
        self.u = u
        self.nx = int(nx)
        self.nt = int(nt)
        
        
        #Other derived parameters
        self.dx = (1./self.nx)
        self.dt = (1./self.nt)
        self.c = self.dt*self.u/self.dx
        
        self.x = np.linspace(0.0, 1.0, self.nx+1)
        self.t = np.zeros(self.nt)
        self.l2 = np.zeros(self.nt)
        
        self.err=errors
        
        self.phi=self.analytic(0)
        self.phiOld=self.phi.copy()
        self.phiOlder=self.phi.copy()
        
        self.figures=plot
        
        
    #the analytical solution to the advection equation, used for comparison and for setting initial conditions
    def analytic(self,t):
        return np.power(np.sin(2*np.pi*(self.x-self.u*t)),2)
    
    
    #function to plot interesting time steps
    def plot_timestep(self,n,scheme):
        if (self.figures):
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
            ax.set_ylim([-0.1,1.1])
            fig.savefig(scheme + "_step%i"%n + ".jpg")
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
            
            #calculating errors to check stability
            self.t[n]=(n+1)*self.dt
            self.l2[n]=self.ltwo(self.phi,self.analytic(self.t[n]),self.dx)
                
            #selecting and plotting 5 timesteps
            y=self.nt/5
            if n%y==0:
                self.plot_timestep(n,scheme)
        
        if (self.err):
            return self.t, self.l2
        else:
            return self.phi
        
                
    #FTCS - computes the numerical solution and calls the plotting function
    def ftcs(self):
        
        scheme='FTCS'
        
        for n in range(self.nt):
            for j in range(1,self.nx):
                self.phi[j] = self.phiOld[j] - self.c*(self.phiOld[j+1] - self.phiOld[j-1])/2
            self.phi[0]=self.phiOld[0]-self.c*(self.phiOld[1]-self.phiOld[-2])/2    
            self.phi[-1]=self.phi[0] 
            self.phiOld = self.phi.copy()
            
            #calculating errors to check stability
            self.t[n]=(n+1)*self.dt
            self.l2[n]=self.ltwo(self.phi,self.analytic(self.t[n]),self.dx)
                
            #selecting and plotting 5 timesteps
            y=self.nt/5
            if n%y==0:
                self.plot_timestep(n,scheme)
        
        if (self.err):
            return self.t, self.l2
        else:
            return self.phi
        
    
    #CTCS - computes the numerical solution and calls the plotting function
    def ctcs(self):
        
        scheme='CTCS'
        self.phiOld=self.analytic(self.dt)
        
        for n in range(1,self.nt):
            for j in range(1,self.nx):
                self.phi[j] = self.phiOlder[j] - self.c*(self.phiOld[j+1] - self.phiOld[j-1])
            self.phi[0]=self.phiOlder[0]-self.c*(self.phiOld[1]-self.phiOld[-2])   
            self.phi[-1]=self.phi[0] 
            self.phiOlder = self.phiOld.copy()
            self.phiOld = self.phi.copy()
            
            #calculating errors to check stability
            self.t[n]=(n+1)*self.dt
            self.l2[n]=self.ltwo(self.phi,self.analytic(self.t[n]),self.dx)
                
            #selecting and plotting 5 timesteps
            y=self.nt/5
            if n%y==0:
                self.plot_timestep(n,scheme)
        
        if (self.err):
            return self.t, self.l2
        else:
            return self.phi
    
    #Root mean square error
    #def rmse(self,res,exp):
        #return np.sqrt(np.mean((exp-res)**2))
    
    
    #L2 norm of errors, used for stability and convergence analysis
    def ltwo(self,res,exp,dx):
        return np.sqrt(np.sum(dx*(res-exp)**2))/np.sqrt(np.sum(dx*exp**2))
    
        