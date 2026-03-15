import numpy as np


def back_substitution(R: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Обратная подстановка для решения R x = y.
    R - верхняя треугольная матрица
    """
    n = R.shape[0]
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-15:
            raise ValueError(f"Нулевой диагональный элемент в позиции {i}")

        s = np.dot(R[i, i + 1:], x[i + 1:]) if i + 1 < n else 0.0
        x[i] = (y[i] - s) / R[i, i]

    return x


def solve_mgs(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решение СЛАУ методом ортогонализации (Modified Gram-Schmidt).
    """
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()

    n = A.shape[0]

    # Проверка на квадратную матрицу
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    # Инициализация Q и R
    Q = np.zeros((n, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    # MGS алгоритм
    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)

        if R[j, j] < 1e-15:
            raise ValueError(f"Матрица вырождена на шаге {j}")

        Q[:, j] = v / R[j, j]

    # Решение R x = Q^T b
    y = Q.T @ b
    return back_substitution(R, y)