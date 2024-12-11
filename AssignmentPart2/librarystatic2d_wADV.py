#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:59:22 2024

@author: arianna
"""

import numpy as np
import matplotlib.pyplot as plt

def shape(xi):
    """
    Returns shape functions given reference coordinates xi in reference triangle

    Parameters
    ----------
    xi: array of floats 2x1
        local coordinates of a point in the reference triangle

    Returns
    -------
    Array of floats 3x1
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
    xi: array of floats 2x1
        local coordinates of a point in the reference triangle
    
    xn: array of floats 3x2
        global coordinates of element nodes

    Returns
    -------
    Array of floats
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
    Array of floats
        Global coordinates of a point in the reference triangle mapped back to the element triangle

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
    

    Parameters
    ----------
    xe: array of floats
        global coordinates of element nodes

    Returns
    -------
    J : array of floats
        2x2 matrix which is the jacobian of the transformation from xi to x

    """

    dN=dNdxi()
    return np.array(xe).T @ dN


def det_j(xe):
    """
    

    Parameters
    ----------
    xe: array of floats
        global coordinates of element nodes

    Returns
    -------
    float
        determinant of jacobian

    """
    J=jacobian(xe)
    return np.linalg.det(J)

    
def inverse_jacobian(xe):
    """
    

    Parameters
    ----------
    xe : array of floats
        global coordinates of element nodes

    Returns
    -------
    float
        inverse of jacobian

    """
    J=jacobian(xe)
    return np.linalg.inv(J)


def dNdx(xe):
    """
    

    Parameters
    ----------
    xe : array of floats
        global coordinates of element nodes

    Returns
    -------
    dNx : Array of floats, Array of floats
        Ordered derivative of shape functions wrt global x,  y.
        

    """
    dN_dxi=dNdxi()
    J_inv=inverse_jacobian(xe)
    dNdx=dN_dxi@J_inv
    return dNdx

def quadrature_over_ref(psi):
    """
    

    Parameters
    ----------
    psi : array of floats
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


def quadrature_over_e(psi,xe):
    """
    

    Parameters
    ----------
    psi : array of floats
        function to integrate over element
        
    xe: array of floats
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
        quad+=psi(x_from_xi(xij[i,:],xe))*abs(det_j(xe))/6
    return quad

        
        
def local_stiff_diff(xe):
    out=np.zeros((3,3))
    dN_dxi = dNdxi()                                      
    dN_dxdy = dN_dxi @ inverse_jacobian(xe)
    for i in range(3):
        for j in range(3):
            psi=lambda x: dN_dxdy[i,0]*dN_dxdy[j,0]+dN_dxdy[i,1]*dN_dxdy[j,1]
            out[i,j]=quadrature_over_e(psi,xe)
    return out

def local_stiff_adv(xe,u):
    out=np.zeros((3,3))
    dN_dxi = dNdxi()                                      
    dN_dxdy = dN_dxi @ inverse_jacobian(xe)
    for i in range(3):
        for j in range(3):
            psi=lambda x: abs(det_j(xe))*shape(x)[i]*(u[0]*dN_dxdy[j,0]+u[1]*dN_dxdy[j,1])
            out[i,j]=quadrature_over_ref(psi)
    return out

def local_stiffness(xe, u, D):
    return D*local_stiff_diff(xe)-local_stiff_adv(xe, u)

def local_force(S,xe):
    f=np.zeros(3)

    for i in range(len(f)):
        integrand = lambda x: abs(det_j(xe))*S(x)*shape(xi_from_x(x,xe))[i]
        f[i]=quadrature_over_e(integrand,xe)
        
    return f

def closest_node(xe):
    Rx=np.array([473993, 171625])
    distance=np.zeros(len(xe[:,0]))
    for i in range(len(distance)):
        distance[i]=np.sqrt((xe[i,0]-Rx[0])**2+(xe[i,1]-Rx[1])**2)
                                                                                                                                                     
    return np.where(distance == distance.min())

def candidates(xe,IEN):
    gnn=closest_node(xe)
    numbers=[]
    for e in range(len(IEN[:,0])):
        if (gnn==IEN[e,0]) or (gnn==IEN[e,1]) or (gnn==IEN[e,2]):
            numbers.append(IEN[e,:])
            
    return np.array(numbers)
    
    
def triangle_finder(xe,IEN,x):
    cands=candidates(xe,IEN)
    
    for e in range(len(cands[:,0])):
        Rxi=xi_from_x(x, xe[cands[e,:],:])
        if (Rxi[0]>=0) and (Rxi[1]>=0) and ((Rxi[0]+Rxi[1])<=1):
            #print("Triangle found! It's candidate number",e+1)
            break
        
    return xe[cands[e,:],:], cands[e,:], Rxi
  
def psi_r(xe,IEN,psi):
    Rx=np.array([473993, 171625])
    tri_coords, tri_num, Rxi = triangle_finder(xe,IEN,Rx)
    psi_tri=psi[tri_num]
    
    N=shape(Rxi)
    psi_R=N[0]*psi_tri[0]+N[1]*psi_tri[1]+N[2]*psi_tri[2]
    
    return psi_R

def psi_s(xe,IEN,psi):
    Sx=np.array([442365, 115483])
    tri_coords, tri_num, Sxi = triangle_finder(xe,IEN,Sx)
    psi_tri=psi[tri_num]
    
    N=shape(Sxi)
    psi_S=N[0]*psi_tri[0]+N[1]*psi_tri[1]+N[2]*psi_tri[2]
    
    return psi_S