from typing import Callable
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from panoptica.science.numerical import solver
from panoptica.science.phys import snells_law
from panoptica.science.linalg import find_angle, unit_vector, init_LGN_vectors

class LightSource:
    """A class that represents a light source
    
    Attributes
    ----------
    density : float
        Number of rays
    K : array_like
        Array of angle coefficients of rays represented as a linear function
    B : array_like
        Array of angle coefficients of rays represented as a linear function
    ray_points : array_like
        Points of ray propagation
    """
    def __init__(self, 
                 position: ArrayLike, 
                 direction: float, 
                 angle:float, 
                 density:float,
                 degrees: bool = False) -> None:
        """Constructs a light source object
        
        Parameters
        ----------
        position : array_like (size 2)
            A position (x, y)
        direction : float
            Angle at which the beam propagates relative to x axis
        angle : float
            Half of a beam angle
        density : float
            Number of rays
        degrees : bool, optional
            If True, all angles are in degrees (Default: False)
        """
        self.density = density
        if degrees:
            self.K = np.tan(np.radians(np.linspace(direction - angle, 
                                                   direction + angle, 
                                                   density)))
        else:
            self.K = np.tan(np.linspace(direction - angle, 
                                        direction + angle, 
                                        density))
        self.B = position[0]*self.K + position[1]*np.ones_like(self.K)
        self.ray_points = np.array([np.tile(position, (density, 1))])
        
class Surface:
    """A class that represents a surface
    
    Attributes
    ----------
    func : function
        A function which defines the surface
    func_prime : function
        A derivative of func
    radius : float
        Radius of a surface from x axis
    n1 : float
        Refractive index before the surface
    n2 : float
        Refractive index after the surface
    """
    def __init__(self, 
                 radius: float, 
                 func: Callable, 
                 func_prime: Callable = None,
                 n1: float = 1,
                 n2: float = 1) -> None:
        """Constructs a surface object
        
        Parameters
        ----------
        radius : float
            Radius of a surface from x axis
        func : function
            A function which defines the surface
        func_prime : function, optional
            A derivative of func (Default: None)
        n1 : float, optional
            Refractive index before the surface (Default: 1)
        n2 : float, optional
            Refractive index after the surface  (Default: 1)
        """
        self.func = func
        self.func_prime = func_prime
        self.radius = radius
        
        self.n1 = n1
        self.n2 = n2
        
class OpticalSystem:
    """A class that represents an optical system with surfaces and light sources
    
    Attributes
    ----------
    light_sources : array_like
        An array of light sources
    surfaces : array_like
        An array of surfaces
    fig : Figure
        Matplotlib figure object
    ax : Axis
        Matplotlib axis object
    """
    def __init__(self) -> None:
        """Constructs an optical system object"""
        self.light_sources = np.empty(0)
        self.surfaces = np.empty(0)
        
    def add_light_sources(self, *light_sources: ArrayLike) -> None:
        """Adds light sources to the optical system
        
        Parameters
        ----------
        light_sources : array_like
            An array of light sources
        """
        self.light_sources = np.append(self.light_sources, light_sources)
        
    def add_surfaces(self, *surfaces):
        """Adds light surfaces to the optical system
        
        Parameters
        ----------
        surfaces : array_like
            An array of surfaces
        """
        self.surfaces = np.append(self.surfaces, sorted(surfaces, 
                                                        key=lambda _: _.func(0),
                                                        reverse=True))
        
    def simulate(self) -> None:
        """Simulates the optical system"""
        for light_source in self.light_sources:
            K_temp = light_source.K
            B_temp = light_source.B
            ray_position = light_source.ray_points[0]
            
            # To stop UnboundLocalError
            last_N = None
            last_G = None
            last_grid = None
            last_func = lambda *args, **kwargs: None
            betas = None
            
            for i, surface in enumerate(self.surfaces):
                grid, conv = solver(K_temp, B_temp, surface.func, surface.func_prime)
                L, G, N = init_LGN_vectors(grid, ray_position, surface.func, surface.func_prime)
                
                mask = np.logical_and(conv, np.abs(grid) < surface.radius)
                grid[~mask] = np.nan
                new_ray_points = np.append(grid.reshape(-1, 1),
                                           surface.func(grid).reshape(-1, 1),
                                           axis=1)
                
                # Builds the non-refractable rays except the first iteration
                if i != 0:
                    last_mask = np.logical_and(~mask, ~np.isnan(last_grid))
                    last_grid = last_grid[last_mask]

                    n1, n2 = -unit_vector(last_N)[last_mask], -unit_vector(last_G)[last_mask]
                    length_after_refraction = 5
                    R = length_after_refraction * (n1 + (np.tan(betas[last_mask]) * n2.T).T)

                    new_ray_points[:, 0][last_mask] = last_grid + R[:, 0]
                    new_ray_points[:, 1][last_mask] = last_func(last_grid) + R[:, 1]
                
                # These are recalculated after the if block so we dont have to seve them
                betas = snells_law(find_angle(L, G) - np.pi / 2,
                                   n1=surface.n1,
                                   n2=surface.n2)
                
                K_temp = np.tan(np.pi/2 + np.arctan(surface.func_prime(grid)) - betas)
                B_temp = surface.func(grid) - grid * K_temp
                
                # These are recalculated at the start of the iteration so we have to save them
                last_N = N
                last_G = G
                last_grid = grid
                last_func = surface.func

                ray_position = new_ray_points
                light_source.ray_points = np.append(light_source.ray_points, 
                                                    [new_ray_points], axis=0)
                
                # Continues the rays after the last surface
                if i == self.surfaces.size - 1:
                    n1, n2 = -unit_vector(N), -unit_vector(G)
                    length_after_refraction_last = 5
                    R = length_after_refraction_last * (n1 + (np.tan(betas) * n2.T).T)
                    
                    last_ray_points = np.append((grid + R[:, 0]).reshape(-1, 1),
                                                (surface.func(last_grid) + R[:, 1]).reshape(-1, 1),
                                                axis=1)
                    
                    light_source.ray_points = np.append(light_source.ray_points, 
                                                        [last_ray_points], axis=0)
                                      
                
                
                
    def show_plot(self) -> None:
        """Shows the optical system using Matplotlib"""
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal', adjustable='box')
        
        for light_source in self.light_sources:
            rays = light_source.ray_points
            for i in range(rays.shape[1]):
                self.ax.plot(rays[:, i, 0], rays[:, i, 1], c='orange')
                
        for surface in self.surfaces:
            x = np.linspace(-surface.radius, surface.radius, 1000)
            self.ax.plot(x, surface.func(x), c='blue')
        
                
        plt.show()