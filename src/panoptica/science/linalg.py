"""Linear algebra for panoptica

Functions
---------
    unit_vector
        Returns the unit vector (array of unit vectors) of the vector (array)
    find_angle
        Returns the smallest angle between v1 and v2
    init_LGN_vectors
        Defining and initialising the line, grad and normal vectors
"""

import numpy as np
from typing import Callable

def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the unit vector (array of unit vectors) of the vector (array).
    
    Parameters
    ----------
    vector : array_like
        A vector or array of vectors to normalise, must be 1D or 2D
        
    Returns
    -------
    norm_vector : array_like
        A normalized vector or array of vectors
    """
    if len(vector.shape) > 2:
        raise IndexError("Maximum vector dimensions is 2")
    
    norm_vector = (vector.T / np.linalg.norm(vector, axis=vector.ndim-1)).T
    
    return norm_vector

def find_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns the smallest angle between v1 and v2
    
    Parameters
    ----------
    v1 : array_like
    v2 : array_like
    
    Returns
    -------
    angle : float
        The smallest angle between two vectors
    """
    
    if v1.shape != v2.shape:
        raise IndexError("Vectors must be the same shape")
    
    if v1.ndim > 2 or v2.ndim > 2:
        raise IndexError("Maximum vector dimension is 2")
    
    # axis to do perform actions on
    ax = v1.ndim - 1
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    angle = np.arccos(np.clip((v1_u*v2_u).sum(ax), -1.0, 1.0))
    
    return angle

def init_LGN_vectors(grid: np.ndarray, origin: np.ndarray, f: Callable, f_prime: Callable):
    '''Defining and initialising the line, grad and normal vectors
    
    Parameters
    ----------
    grid : array_like
    origin : array_like
    f : function
    f_prime : function
    
    Returns
    -------
    L : array_like
    G : array_like
    N : array_like
    '''
    L = np.column_stack((grid, f(grid))) - origin 
    G = np.column_stack((np.ones(len(grid)), f_prime(grid)))
    N = np.column_stack((-f_prime(grid), np.ones(len(grid))))
    
    return L, G, N