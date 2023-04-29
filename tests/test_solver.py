import unittest
import numpy as np
import matplotlib.pyplot as plt
from panoptica.science.numerical import solver
    
class TestSolverFunction(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.K = np.arctan(np.linspace(-5, 5, self.N))
        self.B = np.ones(self.N)*5

    def test_zeros(self):
        
        def linear(x):
            return np.zeros(self.N)
        def linear_prime(x):
            return np.zeros(self.N)
        
        np.testing.assert_almost_equal(solver(self.K, self.B, linear, f_prime=linear_prime)[0],\
                                       (- self.B) / (self.K), decimal=5)
        
    def test_quadratic(self):
        def quad(x):
            return x**2
        def quad_prime(x):
            return 2*x
        
        np.testing.assert_almost_equal(np.abs(solver(self.K, self.B, quad, f_prime=quad_prime)[0]),\
                                       np.minimum(np.abs((self.K + np.sqrt(self.K**2 + 4 * self.B))/(2)),\
                                       np.abs((self.K - np.sqrt(self.K**2 + 4 * self.B))/(2))))
        
if __name__ == "__main__":
    unittest.main()