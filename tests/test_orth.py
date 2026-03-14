import numpy as np
from orth_solver import solve_mgs


def test_mgs_solver():
    A = np.array([
        [10.0, 2.0, 1.0],
        [2.0, 10.0, 1.0],
        [1.0, 1.0, 10.0]
    ])
    x_true = np.array([1.0, 2.0, 3.0])
    b = A @ x_true

    x = solve_mgs(A, b)

    assert np.allclose(x, x_true, atol=1e-8)