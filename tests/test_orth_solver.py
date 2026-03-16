import numpy as np
import pytest

from orth_solver import solve_mgs, back_substitution


def test_back_substitution_3x3():
    """Тест обратной подстановки."""
    R = np.array([
        [4.0, 1.0, 2.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 2.0]
    ])
    y = np.array([12.0, 9.0, 6.0])  # решение: [1, 2, 3]

    x = back_substitution(R, y)

    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(x, expected, atol=1e-10)


def test_back_substitution_invalid_shape():
    """Проверка на неквадратную матрицу R."""
    R = np.array([
        [1.0, 2.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        back_substitution(R, y)


def test_back_substitution_invalid_y_size():
    """Проверка несовпадения размерности y."""
    R = np.array([
        [2.0, 1.0],
        [0.0, 3.0]
    ])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        back_substitution(R, y)


def test_orth_small_system():
    """Решение корректной системы 3x3."""
    A = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 5.0, 1.0],
        [1.0, 1.0, 6.0]
    ])
    b = np.array([9.0, 14.0, 21.0])  # решение: [1, 2, 3]

    x = solve_mgs(A, b)

    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(x, expected, atol=1e-10)


def test_orth_matches_numpy():
    """Сравнение с numpy.linalg.solve на случайной невырожденной матрице."""
    np.random.seed(42)
    n = 5
    A = np.random.rand(n, n)
    A += np.eye(n) * 5.0
    b = np.random.rand(n)

    x = solve_mgs(A, b)
    expected = np.linalg.solve(A, b)

    assert np.allclose(x, expected, atol=1e-10)


def test_orth_singular_matrix():
    """Проверка реакции на вырожденную матрицу."""
    A = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 5.0, 7.0]
    ])
    b = np.array([1.0, 2.0, 3.0])

    with pytest.raises(np.linalg.LinAlgError):
        solve_mgs(A, b)


def test_orth_non_square():
    """Проверка на неквадратную матрицу."""
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    b = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        solve_mgs(A, b)


def test_orth_invalid_b_size():
    """Проверка несовпадения размерности b."""
    A = np.array([
        [2.0, 1.0],
        [1.0, 3.0]
    ])
    b = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        solve_mgs(A, b)