import numpy as np
import time

from gauss_solver import solve_gauss
from orth_solver import solve_mgs


def test_load_solvers_small_benchmark():
    """
    Нагрузочный тест малой размерности.
    Не проверяет строгую скорость, но подтверждает,
    что оба алгоритма корректно работают на матрице увеличенного размера.
    """
    np.random.seed(42)
    n = 100

    A = np.random.rand(n, n)
    A += np.eye(n) * n
    b = np.random.rand(n)

    start = time.perf_counter()
    x_gauss = solve_gauss(A, b)
    gauss_time = time.perf_counter() - start

    start = time.perf_counter()
    x_mgs = solve_mgs(A, b)
    mgs_time = time.perf_counter() - start

    # Проверяем корректность решений
    assert np.allclose(A @ x_gauss, b, atol=1e-6)
    assert np.allclose(A @ x_mgs, b, atol=1e-6)

    # Не делаем жёстких требований по скорости, только sanity-check
    assert gauss_time > 0
    assert mgs_time > 0