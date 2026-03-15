import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gauss_solver import solve_gauss


def test_gauss_small_system():
    """Тест на маленькой системе 3x3"""
    A = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 5.0, 1.0],
        [1.0, 1.0, 6.0]
    ])
    b = np.array([9.0, 14.0, 21.0])  # A * [1,2,3] = [4+2+3, 1+10+3, 1+2+18] = [9,14,21]

    x = solve_gauss(A, b)

    # Проверяем, что решение близко к [1,2,3]
    assert abs(x[0] - 1.0) < 1e-10
    assert abs(x[1] - 2.0) < 1e-10
    assert abs(x[2] - 3.0) < 1e-10