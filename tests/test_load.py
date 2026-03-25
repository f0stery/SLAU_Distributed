import numpy as np
import time

from gauss_solver import solve_gauss
from orth_solver import solve_mgs


def test_load_solvers_small_sanity():
    """
    Лёгкий нагрузочный sanity-check.

    Он только подтверждает, что оба алгоритма:
    1) корректно работают на матрице увеличенного размера;
    2) возвращают решение с малой невязкой;
    3) выполняются за конечное измеримое время.
    """
    print("\n[TEST] test_load_solvers_small_sanity: проверка работы решателей на матрице увеличенного размера", flush=True)

    np.random.seed(42)
    n = 100

    print(f"[INFO] Размер тестовой системы: {n}x{n}", flush=True)

    A = np.random.rand(n, n)
    A += np.eye(n) * n
    b = np.random.rand(n)

    print("[INFO] Матрица A с диагональным преобладанием сформирована", flush=True)

    # Метод Гаусса
    print("[INFO] Запуск solve_gauss(...)", flush=True)
    start = time.perf_counter()
    x_gauss = solve_gauss(A, b)
    gauss_time = time.perf_counter() - start
    print(f"[INFO] Время solve_gauss: {gauss_time:.6f} сек", flush=True)

    # Метод MGS
    print("[INFO] Запуск solve_mgs(...)", flush=True)
    start = time.perf_counter()
    x_mgs = solve_mgs(A, b)
    mgs_time = time.perf_counter() - start
    print(f"[INFO] Время solve_mgs: {mgs_time:.6f} сек", flush=True)

    # Проверка корректности по невязке
    residual_gauss = np.linalg.norm(A @ x_gauss - b)
    residual_mgs = np.linalg.norm(A @ x_mgs - b)

    print(f"[INFO] Невязка метода Гаусса: {residual_gauss:.6e}", flush=True)
    print(f"[INFO] Невязка метода MGS: {residual_mgs:.6e}", flush=True)

    assert np.allclose(A @ x_gauss, b, atol=1e-6)
    print("[OK] Решение solve_gauss корректно", flush=True)

    assert np.allclose(A @ x_mgs, b, atol=1e-6)
    print("[OK] Решение solve_mgs корректно", flush=True)

    # Время должно быть корректно измерено
    assert gauss_time > 0
    assert mgs_time > 0
    print("[OK] Время выполнения обоих методов успешно измерено", flush=True)

    print("[SUCCESS] test_load_solvers_small_sanity успешно пройден", flush=True)