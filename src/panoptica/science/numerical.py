"""Numerical methods for panoptica

Functions
---------
    uniform_grid
        Constructs a uniform grid along the function f(x) with derivative f_prime(x)
    solver
        Finds an intersection point between an object defined by the function f(x) and a ray defined by y = kx + b
"""

from typing import Callable
import warnings as w
import numpy as np
import scipy.optimize as opt

def uniform_grid(length: float, 
                 x_start: float, 
                 x_end: float, 
                 f_prime, 
                 buffer: int = 1000) -> np.ndarray:
    """Constructs a uniform grid along the function f(x) with derivative f_prime(x)
    
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
    """
    grid = np.zeros(buffer)
    grid[0] = x_start
    i=0
    
    while grid[i] < x_end and i < buffer-1:
        grid[i + 1] = grid[i] + length/((1 + f_prime(grid[i])**2)**0.5)
        i+=1
    
    if i == buffer-1:
        w.warn(f"The grid was construced up to x = {grid[i]} \
               because of a small buffer.")
        
    return grid[:i]

def solver(K: np.ndarray, B: np.ndarray, f: Callable, f_prime: Callable = None):
    """Finds an intersection point between an object defined by the function f(x) and a ray defined by y = kx + b
    
    Parameters
    ----------
    K : array_like
        Array of k coefficients in kx + b
    B : array_like
        Array of b coefficients in kx + b
    f : function
        A function that defines the object
    f_prime : function, optional
        A derivative of f(x)
        
    Returns
    -------
    roots : array_like
        An array of roots of f(x) - kx - b = 0
    conv : array_like
        An array that tells if the method converged
    """
    if len(B) != len(K):
        raise ValueError("K and B must be the same size")
    
    # in order to find the intersection point (i. e. f(x) = kx + b) you must solve f(x) - kx - b = 0
    def f_to_solve(x, K, B):
        return f(x) - K*x - B
    
    if f_prime is not None:
        def f_prime_solve(x, K, B):
            return f_prime(x) - K
    
    roots, conv, _ = opt.newton(f_to_solve, np.zeros(len(K)), \
                                args=(K, B), fprime=f_prime_solve, full_output=True)
    
    return roots, conv