import numpy as np


def solve_mgs(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Оптимизированная последовательная версия метода ортогонализации
    (Modified Gram-Schmidt, MGS).

    Решает систему A x = b через QR-разложение:
        A = Q R
        R x = Q^T b
    """
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()

    if A.ndim != 2:
        raise ValueError("Матрица A должна быть двумерной")

    m, n = A.shape
    if m != n:
        raise ValueError("Матрица должна быть квадратной")

    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError("Размер вектора b должен совпадать с размерностью матрицы A")

    Q = np.zeros((n, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    V = A.copy()

    for j in range(n):
        norm = np.linalg.norm(V[:, j])
        if norm < 1e-15:
            raise np.linalg.LinAlgError(f"Матрица вырождена на шаге {j}")

        Q[:, j] = V[:, j] / norm
        R[j, j] = norm

        if j + 1 < n:
            R[j, j + 1:] = Q[:, j] @ V[:, j + 1:]
            V[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])

    y = Q.T @ b
    return back_substitution(R, y)


def back_substitution(R: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Обратная подстановка для решения R x = y,
    где R — верхняя треугольная матрица.
    """
    R = np.asarray(R, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("Матрица R должна быть квадратной")

    n = R.shape[0]

    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("Размер вектора y должен совпадать с размерностью R")

    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-15:
            raise np.linalg.LinAlgError(f"Нулевой диагональный элемент в позиции {i}")

        x[i] = (y[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    return x