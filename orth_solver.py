import numpy as np


def solve_mgs(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Последовательное решение СЛАУ методом ортогонализации
    (Modified Gram-Schmidt / QR-разложение)
    """
    A = A.astype(np.float64).copy()
    b = b.astype(np.float64).copy()

    n = A.shape[0]
    Q = np.zeros_like(A)
    R = np.zeros((n, n), dtype=np.float64)

    V = A.copy()

    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        if R[j, j] < 1e-12:
            raise ValueError("Линейно зависимые столбцы, разложение невозможно")

        Q[:, j] = V[:, j] / R[j, j]

        for k in range(j + 1, n):
            R[j, k] = np.dot(Q[:, j], V[:, k])
            V[:, k] = V[:, k] - R[j, k] * Q[:, j]

    y = Q.T @ b

    # Обратная подстановка
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-12:
            raise ValueError("Нулевой диагональный элемент в R")
        x[i] = (y[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    return x