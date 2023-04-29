import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from panoptica.science.phys import snells_law
from panoptica.science.numerical import solver
from panoptica.science.linalg import find_angle, unit_vector

def f(x):
    return x**2

def f_prime(x):
    return 2*(x)

def init_vectors(grid: np.ndarray, origin: np.ndarray, f: Callable, f_prime: Callable):
    '''Defining and initialising the line, grad and normal vectors'''
    L = np.column_stack((grid, f(grid))) - origin 
    G = np.column_stack((np.ones(len(grid)), f_prime(grid)))
    N = np.column_stack((-f_prime(grid), np.ones(len(grid))))
    
    return L, G, N

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    
    N = 10
    K = np.tan(np.radians(np.linspace(0.1, 90, N)))
    B = np.ones(N)*5
    grid = solver(K, B, f, f_prime=f_prime)[0]
    print(grid)

    light_source = np.array((0, 5))
    const = 2

    L, G, N = init_vectors(grid, light_source, f, f_prime)  

    n1, n2 = -unit_vector(N), -unit_vector(G)
    beta = snells_law(find_angle(L, G) - np.pi / 2, n2=1.5)

    R = const * n1 + const * (np.tan(beta) * n2.T).T

    for i, x in enumerate(grid):
        ax.plot((light_source[0], x), (light_source[1], f(x)), c='red')
        ax.plot((x, x + R[i, 0]), (f(x), f(x) + R[i, 1]), c='red')
        ax.plot((x, x + N[i, 0]), (f(x), f(x) + N[i, 1]), '--', c='gray')
        ax.plot((x, x + -N[i, 0]), (f(x), f(x) + -N[i, 1]), '--', c='gray')


    x = np.linspace(-2.5, 2.52, 10000)

    ax.plot(x, f(x))
    plt.show()
