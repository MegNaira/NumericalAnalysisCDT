#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:18:23 2024

@author: arianna
"""

import numpy as np
import matplotlib.pyplot as plt

#nodes=np.array([[[0.,1.,0.],
#                 [0.,0.,1.]],
#                
#                [[1.,1.,0.],
#                 [0.,1.,1.]]])

nodes=np.array([[[0.,0.],
                 [1.,0.],
                 [0.,1.]],
                
                [[1.,0.],
                 [1.,1],
                 [0.,1.]]])
n=nodes[0,:,0]
node0_0=nodes[0][0]
node1_0=nodes[0][1]
node2_0=nodes[0][2]
#print(n)


def shape():
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
    xi=np.array([[0.,0.],
                 [1.,0.],
                 [0.,1.]])
    
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
    Array of floats, Array of floats
        Ordered derivative of shape functions wrt xi[0],  xi[1].

    """
    return np.array([-1.,1.,0.]),np.array([-1.,0.,1.])


def x_from_xi():
    """
    Returns the global coordinates of a point xi on the reference triangle, given the global coordinates of the element true nodes

    Parameters
    ----------
    None

    Returns
    -------
    Array of floats
        Global coordinates of a point in the reference triangle mapped back to the element triangle

    """
    xi=np.array([[0.,0.],
                 [1.,0.],
                 [0.,1.]])
    N=shape()
    x=N[0]*xi[0,0]+N[1]*xi[1,0]+N[2]*xi[2,0]
    y=N[0]*xi[0,1]+N[1]*xi[1,1]+N[2]*xi[2,1]
    return np.array([x,y])

def jacobian():
    """
    

    Parameters
    ----------
    None

    Returns
    -------
    J : array of floats
        2x2 matrix which is the jacobian of the transformation from xi to x

    """
    xi=np.array([[0.,0.],
                 [1.,0.],
                 [0.,1.]])
    dN1, dN2=dNdxi()
    J=np.array([[0,0],
                [0,0]])
    for i in range(3):
        J[0][0]+= dN1[i]*xi[i,0]
        J[0][1]+= dN2[i]*xi[i,0]
        J[1][0]+= dN1[i]*xi[i,1]
        J[1][1]+= dN2[i]*xi[i,1]
    return J

def det_j():
    """
    

    Parameters
    ----------
    None

    Returns
    -------
    float
        determinant of jacobian

    """
    J=jacobian()
    return np.linalg.det(J)

    
def test_jacobian():
    #default
    xe=np.array([0,0],[0,1],[1,0])
    ans= np.array([0,1],
                  [1,0])
    assert np.allclose(jacobian(xe),ans)
    
    #translated
    xe=np.array([1,0],[1,1],[2,0])
    ans= np.array([0,1],
                  [1,0])
    assert np.allclose(jacobian(xe),ans)
    
    #scaled
    xe=np.array([0,0],[0,2],[2,0])
    ans= np.array([0,2],
                  [2,0])
    assert np.allclose(jacobian(xe),ans)
    
    #rotated
    xe=np.array([1,1],[0,1],[1,0])
    ans= np.array([-1,0],
                  [0,-1])
    assert np.allclose(jacobian(xe),ans)
    
def inverse_jacobian():
    J=jacobian()
    return np.linalg.inv(J)

def dNdx():
    dN1, dN2=dNdxi()
    J=jacobian()
    J_inv=inverse_jacobian()
    dNx=np.dot(J_inv,np.array([dN1,dN2]))
    return dNx

def quadrature_over_ref(psi):
    quad=0
    
    xi=np.array([[0.,0.],
                 [1.,0.],
                 [0.,1.]])
    
    xij=np.array([1.,1], 
                 [4.,1.],
                 [1.,4.])/6
    
    for i in range(3):
        quad+=psi(x_from_xi(xij[i,:]))
    
    
    
    
if __name__ == "__main__":
    #some test
    print(dNdx())