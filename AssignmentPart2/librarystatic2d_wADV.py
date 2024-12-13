#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:59:22 2024

@author: arianna

Arrays dimensions are expressed as NRxNC where NR = number of rows and NC = number of columns
"""

import numpy as np
import matplotlib.pyplot as plt

def shape(xi):
    """
    Returns shape functions given reference coordinates xi in reference triangle

    Parameters
    ----------
    xi: array of floats 1x2
        local coordinates of a point in the reference triangle

    Returns
    -------
    Array of floats 1x3
        Ordered shape function on points of reference triangle.

    """
    
    N0=1-xi[0]-xi[1]
    N1=xi[0]
    N2=xi[1]
    return np.array([N0,N1,N2])

def dNdxi():
    """
    Returns derivatives of shape functions with respect to xi in reference triangle

    Parameters
    ----------
    None

    Returns
    -------
    Array of floats 3x2
        Ordered derivative of shape functions wrt xi[0],  xi[1].

    """
    return np.array([[-1.,-1.],[1.,0.],[0.,1.]])

def x_from_xi(xi, xn):
    """
    Returns the global coordinates of a point xi on the reference triangle, given the global coordinates of the element nodes

    Parameters
    ----------
    xi: array of floats 1x2
        local coordinates of a point in the reference triangle
    
    xn: array of floats 3x2
        global coordinates of element nodes

    Returns
    -------
    Array of floats 1x2
        Global coordinates of a point in the reference triangle mapped back to the element triangle

    """

    N=shape(xi)
    x=N[0]*xn[0,0]+N[1]*xn[1,0]+N[2]*xn[2,0]
    y=N[0]*xn[0,1]+N[1]*xn[1,1]+N[2]*xn[2,1]
    return np.array([x,y])

def xi_from_x(x, xn):
    """
    Returns the local coordinates of a point x on the element triangle, given the global coordinates of the element nodes

    Parameters
    ----------
    xn: array of floats 3x2
        global coordinates of element nodes
    

    Returns
    -------
    Array of floats 1x2
        Local coordinates of a point in element triangle mapped back to the reference triangle

    """
    A=(x[0]-xn[0,0])/(xn[1,0]-xn[0,0])
    B=(xn[2,0]-xn[0,0])/(xn[1,0]-xn[0,0])
    C=(x[1]-xn[0,1])/(xn[2,1]-xn[0,1])
    D=(xn[1,1]-xn[0,1])/(xn[2,1]-xn[0,1])
    xi1=(A-B*C)/(1-B*D)
    xi2=C-D*((A-B*C)/(1-B*D))
    return np.array([xi1,xi2])

def jacobian(xe):
    """
    Computes the jacobian for the change of coordinates from reference to global

    Parameters
    ----------
    xe: array of floats Nx2
        global coordinates of N element nodes

    Returns
    -------
    array of floats 2x2
        matrix which is the jacobian of the transformation from xi to x

    """

    dN=dNdxi()
    return np.array(xe).T @ dN


def det_j(xe):
    """
    Computes the determinant of the jacobian for the change of coordinates from reference to global

    Parameters
    ----------
    xe: array of floats Nx2
        global coordinates of N element nodes

    Returns
    -------
    float
        determinant of jacobian

    """
    J=jacobian(xe)
    return np.linalg.det(J)

    
def inverse_jacobian(xe):
    """
    Computes the inverse of the jacobian for the change of coordinates from reference to global
    i.e. the jacobian for the change from global to reference coordinates

    Parameters
    ----------
    xe : array of floats Nx2
        global coordinates of N element nodes

    Returns
    -------
    array of floats 2x2
        inverse of jacobian

    """
    J=jacobian(xe)
    return np.linalg.inv(J)


def dNdx(xe):
    """
    Computes the derivatives of the shape function wrt to the global coordinates

    Parameters
    ----------
    xe : array of floats Nx2
        global coordinates of N element nodes

    Returns
    -------
    dNx : Array of floats 3x2
        Ordered derivative of shape functions wrt global x,  y.
        

    """
    dN_dxi=dNdxi()
    J_inv=inverse_jacobian(xe)
    dNdx=dN_dxi@J_inv
    return dNdx

def quadrature_over_ref(psi):
    """
    Computes the gaussian quadrature of the function psi over the reference triangle

    Parameters
    ----------
    psi : func 1d
        function to integrate over reference triangle

    Returns
    -------
    quad : float
        gaussian quadrature of psi over reference triangle

    """
    quad=0
    
    xi=np.array([[0.,0.],
                 [1.,0.],
                 [0.,1.]])
    
    xij=np.array([[1.,1], 
                 [4.,1.],
                 [1.,4.]])/6
    
    for i in range(3):
        quad+=psi(xij[i,:])/6
        
    return quad


def quadrature_over_e(psi,xn):
    """
    Computes the gaussian quadrature of the function psi over the element with nodes xn
    

    Parameters
    ----------
    psi : func 1d
        function to integrate over element
        
    xn: array of floats 3x2
        global coordinates of element nodes

    Returns
    -------
    quad : float
        gaussian quadrature of psi over element

    """
    quad=0
    
    xij=np.array([[1.,1], 
                 [4.,1.],
                 [1.,4.]])/6
    
    
    for i in range(3):
        quad+=psi(x_from_xi(xij[i,:],xn))*abs(det_j(xn))/6
    return quad

        
        
def local_stiff_diff(xe):
    """
    Computes the local stiffness matrix for the diffusion dominated equation

    Parameters
    ----------
    xe : array of floats Nx2
        global coordinates of N element nodes

    Returns
    -------
    out : array of floats 3x3
        local stiffness matrix for diffusion

    """
    out=np.zeros((3,3))
    dN_dxi = dNdxi()                                      
    dN_dxdy = dN_dxi @ inverse_jacobian(xe)
    for i in range(3):
        for j in range(3):
            psi=lambda x: dN_dxdy[i,0]*dN_dxdy[j,0]+dN_dxdy[i,1]*dN_dxdy[j,1]
            out[i,j]=quadrature_over_e(psi,xe)
    return out

def local_stiff_adv(xe,u):
    """
    Computes the local stiffness matrix for the advection dominated equations

    Parameters
    ----------
    xe : array of floats Nx2
        global coordinates of N element nodes
    
    u : array of floats 1x2
        advection velocity in the (x,y) plane

    Returns
    -------
    out : array of floats 3x3
        local stiffness matrix for advection

    """
    out=np.zeros((3,3))
    dN_dxi = dNdxi()                                      
    dN_dxdy = dN_dxi @ inverse_jacobian(xe)
    for i in range(3):
        for j in range(3):
            psi=lambda x: abs(det_j(xe))*shape(x)[i]*(u[0]*dN_dxdy[j,0]+u[1]*dN_dxdy[j,1])
            out[i,j]=quadrature_over_ref(psi)
    return out

def local_stiffness(xe, u, D):
    """
    Sums the advection dominated and diffusion dominated local stiffness matrixes

    Parameters
    ----------
    xe : array of floats Nx2
        global coordinates of N element nodes
        
    u : array of floats 1x2
        advection velocity in the (x,y) plane
        
    D : float
        diffusion coefficient

    Returns
    -------
    array of floats 3x3
        local stiffness matrix for advection+diffusion

    """
    return D*local_stiff_diff(xe)-local_stiff_adv(xe, u)

def local_force(S,xe):
    """
    Computes the column force vector

    Parameters
    ----------
    S : func
        source function
        
    xe : array of floats Nx2
        global coordinates of N element nodes

    Returns
    -------
    f : array of floats 1x3
        force vector

    """
    f=np.zeros(3)

    for i in range(len(f)):
        integrand = lambda x: abs(det_j(xe))*S(x)*shape(xi_from_x(x,xe))[i]
        f[i]=quadrature_over_e(integrand,xe)
        
    return f

def closest_node(xe,x):
    """
    Finds the closest node to the point x in the given grid xe 

    Parameters
    ----------
    xe : array of floats Nx2
        global coordinates of N element nodes
    
    x : array of floats 1x2
        global coordinates of the point of interest

    Returns
    -------
    float
        IEN of the node closest to the point of interest

    """
    
    distance=np.zeros(len(xe[:,0]))
    for i in range(len(distance)):
        distance[i]=np.sqrt((xe[i,0]-x[0])**2+(xe[i,1]-x[1])**2)
                                                                                                                                                     
    return np.where(distance == distance.min())

def candidates(xe,IEN,x):
    """
    Finds the elements that contain the closest node to the point x

    Parameters
    ----------
    xe : array of floats Nx2
        global coordinates of N element nodes
        
    IEN : array of floats Nex3
        IEN of the Ne elements
        
    x : array of floats 1x2
        global coordinates of the point of interest

    Returns
    -------
    array of floats 6x3
        IEN of the 6 elements that contain the closest node to x

    """
    gnn=closest_node(xe,x)
    numbers=[]
    for e in range(len(IEN[:,0])):
        if (gnn==IEN[e,0]) or (gnn==IEN[e,1]) or (gnn==IEN[e,2]):
            numbers.append(IEN[e,:])
            
    return np.array(numbers)
    
    
def triangle_finder(xe,IEN,x):
    """
    Checks on the reference coordinates of the point x to find the element where it sits

    Parameters
    ----------
    xe : array of floats Nx2
        global coordinates of N element nodes
        
    IEN : array of floats Nex3
        IEN of the Ne elements
        
    x : array of floats 1x2
        global coordinates of the point of interest

    Returns
    -------
    array of floats 3x2
        global coordinates of the 3 nodes in the element that contains x
        
    array of floats 1x3
        IEN of the 3 nodes in the element that contains x
        
    Pxi : array of floats 1x2
        reference coordinates of the point x

    """
    cands=candidates(xe,IEN,x)
    
    for e in range(len(cands[:,0])):
        Pxi=xi_from_x(x, xe[cands[e,:],:])
        if (Pxi[0]>=0) and (Pxi[1]>=0) and ((Pxi[0]+Pxi[1])<=1):
            break
        
    return xe[cands[e,:],:], cands[e,:], Pxi
  
def psi_p(xe,IEN,psi,x):
    """
    Evaluates the value of psi in Reading (global coords given by Rx) using shape functions

    Parameters
    ----------
    xe : array of floats Nx2
        global coordinates of N element nodes
        
    IEN : array of floats Nex3
        IEN of the Ne elements
        
    psi : func 1d
        function to integrate over element
    
    x : array of floats 1x2
        global coordinates of the point of interest P

    Returns
    -------
    psi_P : float
        value of psi in point P

    """
    #Rx=np.array([473993, 171625])
    tri_coords, tri_num, Pxi = triangle_finder(xe,IEN,x)
    psi_tri=psi[tri_num]
    
    N=shape(Pxi)
    psi_P=N[0]*psi_tri[0]+N[1]*psi_tri[1]+N[2]*psi_tri[2]
    
    return psi_P
