import numpy as np


def solve_gauss(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решение СЛАУ методом Гаусса с частичным выбором главного элемента.

    Решает систему:
        A x = b

    Используется для сравнения с методом ортогонализации (MGS).
    """
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).copy()

    if A.ndim != 2:
        raise ValueError("Матрица A должна быть двумерной")

    m, n = A.shape
    if m != n:
        raise ValueError("Матрица A должна быть квадратной")

    if b.ndim != 1 or b.shape[0] != n:
        raise ValueError("Размер вектора b должен совпадать с размерностью матрицы A")

    # Прямой ход с частичным выбором главного элемента
    for k in range(n):
        pivot = np.argmax(np.abs(A[k:, k])) + k

        if abs(A[pivot, k]) < 1e-12:
            raise np.linalg.LinAlgError("Матрица вырождена или близка к вырожденной")

        # Перестановка строк
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]

        # Обнуление элементов ниже диагонали
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Обратный ход
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < 1e-12:
            raise np.linalg.LinAlgError("Нулевой элемент на диагонали при обратном ходе")

        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x