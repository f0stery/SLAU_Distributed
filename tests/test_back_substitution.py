import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orth_solver import back_substitution


def test_back_substitution_3x3():
    """Тест обратной подстановки"""
    R = np.array([
        [4.0, 1.0, 2.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 2.0]
    ])
    # R * [1,2,3] = [4*1 + 1*2 + 2*3, 3*2 + 1*3, 2*3] = [4+2+6, 6+3, 6] = [12, 9, 6]
    y = np.array([12.0, 9.0, 6.0])

    x = back_substitution(R, y)

    assert abs(x[0] - 1.0) < 1e-10
    assert abs(x[1] - 2.0) < 1e-10
    assert abs(x[2] - 3.0) < 1e-10