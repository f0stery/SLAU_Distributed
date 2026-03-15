import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orth_solver import solve_mgs


def test_orth_small_system():
    """Тест на маленькой системе 3x3"""
    A = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 5.0, 1.0],
        [1.0, 1.0, 6.0]
    ])
    b = np.array([9.0, 14.0, 21.0])  # A * [1,2,3]

    x = solve_mgs(A, b)

    assert abs(x[0] - 1.0) < 1e-10
    assert abs(x[1] - 2.0) < 1e-10
    assert abs(x[2] - 3.0) < 1e-10


def test_orth_singular_matrix():
    """Тест на вырожденной матрице"""
    A = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [3, 5, 7]
    ])
    b = np.array([1, 2, 3])

    try:
        solve_mgs(A, b)
        assert False, "Должна быть ошибка"
    except (ValueError, np.linalg.LinAlgError):
        pass


def test_orth_non_square():
    """Тест на неквадратной матрице"""
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    b = np.array([1.0, 2.0, 3.0])

    try:
        solve_mgs(A, b)
        assert False, "Должна быть ошибка"
    except ValueError:
        pass