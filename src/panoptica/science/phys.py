'''#TODO'''
import numpy as np

def snells_law(alpha: float, n1=1., n2=1.5):
    '''Returns the angle of the refracted ray according to the Snell's law
    
    Parameters
    ----------
    alpha : float
        Angle of incidence
    n1 : float (default: 1.)
        Refractive index of the first medium
    n2 : float (default: 1.5)
        Refractive index of the first medium
        
    Returns
    -------
    beta : float
        Angle of refraction
    '''
    beta = np.arcsin(n1*np.sin(alpha) / n2)
    
    return beta