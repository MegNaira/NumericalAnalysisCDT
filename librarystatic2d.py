#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:48:48 2024

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

def test_jacobian():
    #default
    xe=np.array([[0,0],[1,0],[0,1]])
    ans= np.array([[1,0],
                  [0,1]])
    assert np.allclose(jacobian(xe),ans)
    
    #translated
    xe=np.array([[1,0],[2,0],[1,1]])
    ans= np.array([[1,0],
                  [0,1]])
    assert np.allclose(jacobian(xe),ans)
    
    #scaled
    xe=np.array([[0,0],[2,0],[0,2]])
    ans= np.array([[2,0],
                  [0,2]])
    assert np.allclose(jacobian(xe),ans)
    
    #rotated
    xe=np.array([[0,0],[0,1],[-1,0]])
    ans= np.array([[0,-1],
                  [1,0]])
    assert np.allclose(jacobian(xe),ans)

    
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
        quad+=psi(x_from_xi(xij[i,:],xi))/6
        
    return quad

def test_quad_ref():
    
    constant = {
             "psi": lambda x: 1,
             "ans": 0.5 
             } 
    
    linearx = {
             "psi": lambda x: 6*x[0],
             "ans": 1,
             }
    
    lineary = {
             "psi": lambda x: x[1],
             "ans": 1/6,
             }
    
    product = {
             "psi": lambda x: x[0]*x[1],
             "ans": 1/24,
             }
    for t in [constant, linearx, lineary, product]:
        assert np.allclose(quadrature_over_ref(t["psi"]),t["ans"]), f"function\n {t['psi']} is broken"

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


def test_quad_e():   
    translated_linear = {
             "xe": np.array([[1,0],[2,0],[1,1]]),
             "phi": lambda x: 3*x[0],
             "ans": 2,
                        }
    
    translated_product = {
             "xe": np.array([[1,0],[2,0],[1,1]]),
             "phi": lambda x: x[0]*x[1],
             "ans": 5/24,
                         }
    
    scaled_linear = {
             "xe": np.array([[0,0],[2,0],[0,2]]),
             "phi": lambda x: 3*x[0],
             "ans": 4,
                     }
    
    scaled_product = {
             "xe": np.array([[0,0],[2,0],[0,2]]),
             "phi": lambda x: x[0]*x[1],
             "ans": 2/3,
                     }
    
    rotated_linear = {
             "xe": np.array([[0,0],[0,1],[-1,0]]),
             "phi": lambda x: 3*x[0],
             "ans": -1/2,
                     }
    
    rotated_product = {
             "xe": np.array([[0,0],[0,1],[-1,0]]),
             "phi": lambda x: x[0]*x[1],
             "ans": 1/24,
                     }
    
    
    for t in [translated_linear, translated_product, scaled_linear, scaled_product,
              rotated_linear]:
        assert np.allclose(quadrature_over_e(t["phi"], t["xe"]),t["ans"]), f"element\n {t['xe']} and/or function {t['phi']} is broken"
        
        
def local_stiffness(xe):
    dN_dxi = dNdxi()
    detJ = det_j(xe)                          
    J_inv = inverse_jacobian(xe)                     
    dN_dxdy = dN_dxi @ J_inv  
    return  1/2* abs(detJ) * (dN_dxdy @ dN_dxdy.T)


def test_stiffness():
#answers need to be recalcuated
    default = {
                "xe": np.array([[0,0],[1,0],[0,1]]),
                "ans": np.array([[1, -0.5, -0.5],
                                 [-0.5, 0.5, 0],
                                 [-0.5, 0, 0.5]])
              }
    
    translated = {
                "xe": np.array([[1, 1, 2],
                                [0, 1, 0]]),
                "ans": np.array([[1, -0.5, -0.5],
                                 [-0.5, 0.5, 0],
                                 [-0.5, 0, 0.5]])
                 }
    
    scaled = {
                "xe": np.array([[0,0],[2,0],[0,2]]),
                "ans": np.array([[1, -0.5, -0.5],
                                 [-0.5, 0.5, 0],
                                 [-0.5, 0, 0.5]])
            }
    rotated = {
                "xe": np.array([[0,0],[0,1],[-1,0]]),
                "ans": np.array([[1, -0.5, -0.5],
                                 [-0.5, 0.5, 0],
                                 [-0.5, 0, 0.5]])
              }
    
    for t in [default, translated, scaled, rotated]:
        assert np.allclose(local_stiffness(t["xe"].T),t["ans"]), f"element\n {t['xe']} is broken"

def local_force(S,xe):
    f=np.zeros(3)
    detJ=det_j(xe)

    for i in range(len(f)):
        integrand = lambda xi: abs(detJ)*S(x_from_xi(xi, xe))*shape(xi)[i]
        f[i]=quadrature_over_ref(integrand)
        
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
    