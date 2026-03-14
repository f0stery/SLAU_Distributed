import numpy as np


def solve_gauss(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = A.astype(np.float64).copy()
    b = b.astype(np.float64).copy()

    n = len(b)

    # Прямой ход с частичным выбором главного элемента
    for k in range(n):
        pivot = np.argmax(np.abs(A[k:, k])) + k
        if abs(A[pivot, k]) < 1e-12:
            raise ValueError("Матрица вырождена или близка к вырожденной")

        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Обратный ход
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < 1e-12:
            raise ValueError("Нулевой элемент на диагонали при обратном ходе")
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x