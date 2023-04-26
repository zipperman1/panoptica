"""Numerical methods for panoptica

Functions
---------
    uniform_grid
        Constructs a uniform grid along the function f(x) with derivative f_prime(x)
"""

from typing import Callable
import warnings as w
import numpy as np

def uniform_grid(length: float, 
                 x_start: float, 
                 x_end: float, 
                 f_prime, 
                 buffer: int = 1000) -> np.ndarray:
    '''Constructs a uniform grid along the function f(x) with derivative f_prime(x)
    
    Parameters
    ----------
    length : float
        Length of a segment of a function
    x_start : float
        Starting point for the grid
    x_end : float
        Ending point for the grid
    f_prime : function
        The derivative of function f
    buffer : float, optional
        Maximum amount of grid points
        
    Returns
    -------
    grid : array_like
        An array of size < buffer that consists of points on the grid
    '''
    grid = np.zeros(buffer)
    grid[0] = x_start
    i=0
    
    while grid[i] < x_end and i < buffer-1:
        grid[i + 1] = grid[i] + length/((1 + f_prime(grid[i])**2)**0.5)
        i+=1
    
    if i == buffer-1:
        w.warn(f"The grid was construced up to x = {grid[i]} because of a small buffer.")
        
    return grid[:i]
