import numpy as np
from utils import reduced_problem_solver
def test_reduced_problem():
    alphas= np.array([[1, 2]])
    betas = np.array([1])
    C = 1.0
    M=2
    expected_w = np.array([-.2, -.4])
    expected_xi = 0
    w, xi = reduced_problem_solver(alphas, betas, C, M)
    np.testing.assert_array_almost_equal(w, expected_w, decimal=6)
    np.testing.assert_almost_equal(xi, expected_xi, decimal=6)

