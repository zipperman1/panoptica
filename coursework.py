import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def f(x):
    return np.zeros_like(x)

def f_prime(x):
    return np.zeros_like(x)

def grid_maker(length: float , x_start: float , x_end: float, f: Callable, f_prime: Callable, buffer: int = 1000000) -> np.ndarray:
    '''Makes a uniform grid along the function f(x)'''
    grid = np.zeros(buffer)
    grid[0] = x_start
    i=0
    
    while grid[i] < x_end:
        grid[i + 1] = grid[i] + length/((1 + f_prime(grid[i])**2)**0.5)
        i+=1
    
    return grid[:i]

def init_vectors(grid: np.ndarray, origin: np.ndarray, f: Callable, f_prime: Callable):
    '''Defining and initialising the line, grad and normal vectors'''
    L = np.column_stack((grid, f(grid))) - origin 
    G = np.column_stack((np.ones(len(grid)), f_prime(grid)))
    N = np.column_stack((-f_prime(grid), np.ones(len(grid))))
    
    return L, G, N

def unit_vector(vector):
    '''Returns the unit vector (array of unit vectors) of the vector (array).'''
    return (vector.T / np.linalg.norm(vector, axis=1)).T

def find_angle(v1, v2):
    '''Returns an angle between v1 and v2'''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    # i dont know what the fuck am i doing
    return np.arccos(np.clip(np.tensordot(v1_u, v2_u, [1, 1]).diagonal(), -1.0, 1.0)) # diagonal is bad, find out abot tensordot

def snells_law(alpha, n1=1, n2=1.5):
    '''Returns the angle of the refracted ray according to the Snell's law'''
    return np.arcsin(n1*np.sin(alpha) / n2)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    grid = grid_maker(0.5, -2, 2.1, f, f_prime)
    light_source = np.array((0, 2))
    const = 2

    L, G, N = init_vectors(grid, light_source, f, f_prime)  

    n1, n2 = -unit_vector(N), -unit_vector(G)
    beta = snells_law(find_angle(L, G) - np.pi / 2, n2=1.5)

    R = const * n1 + const * (np.tan(beta) * n2.T).T
    print(find_angle(R, -N)* 180 / np.pi)

    for i, x in enumerate(grid):
        ax.plot((light_source[0], x), (light_source[1], f(x)), c='red')
        ax.plot((x, x + R[i, 0]), (f(x), f(x) + R[i, 1]), c='red')
        ax.plot((x, x + N[i, 0]), (f(x), f(x) + N[i, 1]), '--', c='gray')
        ax.plot((x, x + -N[i, 0]), (f(x), f(x) + -N[i, 1]), '--', c='gray')


    x = np.linspace(-2, 2.02, 1000)

    ax.plot(x, f(x))
    plt.show()

