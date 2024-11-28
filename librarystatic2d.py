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
    xi : array containing the coordinates of a point in the reference triangle.

    Returns
    -------
    Array of floats
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
    Array of floats
        Ordered derivative of shape functions wrt xi[0],  xi[1].

    """
    return np.array([[-1.,-1.],[1.,0.],[0.,1.]])

def x_from_xi(xi, xe):
    """
    Returns the global coordinates of a point xi on the reference triangle, given the global coordinates of the element nodes

    Parameters
    ----------
    xe: array of floats
        global coordinates of element nodes

    Returns
    -------
    Array of floats
        Global coordinates of a point in the reference triangle mapped back to the element triangle

    """

    N=shape(xi)
    x=N[0]*xe[0,0]+N[1]*xe[1,0]+N[2]*xe[2,0]
    y=N[0]*xe[0,1]+N[1]*xe[1,1]+N[2]*xe[2,1]
    return np.array([x,y])

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