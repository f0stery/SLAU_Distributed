import numpy as np


def back_substitution(R: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = R.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = np.dot(R[i, i + 1:], x[i + 1:])
        x[i] = (y[i] - s) / R[i, i]
    return x


def solve_mgs(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64)

    m, n = A.shape
    if m != n:
        raise ValueError("Ожидается квадратная матрица A")

    R = np.zeros((n, n), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for k in range(n):
        col = A[:, k]
        norm = np.linalg.norm(col)
        if norm < 1e-15:
            raise np.linalg.LinAlgError(f"Матрица вырождена или почти вырождена на шаге k={k}")

        R[k, k] = norm
        q = col / norm

        y[k] = np.dot(q, b)

        if k + 1 < n:
            r_tail = q @ A[:, (k + 1):]
            R[k, (k + 1):] = r_tail
            A[:, (k + 1):] -= np.outer(q, r_tail)

    return back_substitution(R, y)