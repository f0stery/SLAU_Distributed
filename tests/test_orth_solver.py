import numpy as np
import pytest

from orth_solver import solve_mgs, back_substitution


def test_back_substitution_3x3():
    """Тест обратной подстановки."""
    print("\n[TEST] test_back_substitution_3x3: проверка обратной подстановки для верхнетреугольной матрицы 3x3", flush=True)

    R = np.array([
        [4.0, 1.0, 2.0],
        [0.0, 3.0, 1.0],
        [0.0, 0.0, 2.0]
    ])
    y = np.array([12.0, 9.0, 6.0])  # решение: [1, 2, 3]

    print(f"[INFO] Матрица R:\n{R}", flush=True)
    print(f"[INFO] Вектор y: {y}", flush=True)

    x = back_substitution(R, y)

    print(f"[INFO] Вычисленное решение x: {x}", flush=True)

    expected = np.array([1.0, 2.0, 3.0])
    print(f"[INFO] Ожидаемое решение: {expected}", flush=True)

    assert np.allclose(x, expected, atol=1e-10)
    print("[SUCCESS] test_back_substitution_3x3 успешно пройден", flush=True)


def test_back_substitution_invalid_shape():
    """Проверка на неквадратную матрицу R."""
    print("\n[TEST] test_back_substitution_invalid_shape: проверка обработки неквадратной матрицы R", flush=True)

    R = np.array([
        [1.0, 2.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    y = np.array([1.0, 2.0, 3.0])

    print(f"[INFO] Некорректная матрица R формы {R.shape}:\n{R}", flush=True)
    print(f"[INFO] Вектор y: {y}", flush=True)
    print("[INFO] Ожидается исключение ValueError", flush=True)

    with pytest.raises(ValueError):
        back_substitution(R, y)

    print("[SUCCESS] ValueError корректно выброшено для неквадратной матрицы R", flush=True)


def test_back_substitution_invalid_y_size():
    """Проверка несовпадения размерности y."""
    print("\n[TEST] test_back_substitution_invalid_y_size: проверка несовпадения размерности y", flush=True)

    R = np.array([
        [2.0, 1.0],
        [0.0, 3.0]
    ])
    y = np.array([1.0, 2.0, 3.0])

    print(f"[INFO] Матрица R формы {R.shape}:\n{R}", flush=True)
    print(f"[INFO] Некорректный вектор y длины {len(y)}: {y}", flush=True)
    print("[INFO] Ожидается исключение ValueError", flush=True)

    with pytest.raises(ValueError):
        back_substitution(R, y)

    print("[SUCCESS] ValueError корректно выброшено при несовпадении размерности y", flush=True)


def test_orth_small_system():
    """Решение корректной системы 3x3."""
    print("\n[TEST] test_orth_small_system: решение корректной системы 3x3 методом Модифицированного Грама–Шмидта", flush=True)

    A = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 5.0, 1.0],
        [1.0, 1.0, 6.0]
    ])
    b = np.array([9.0, 14.0, 21.0])  # корректное решение: [1, 2, 3]

    print(f"[INFO] Матрица A:\n{A}", flush=True)
    print(f"[INFO] Вектор b: {b}", flush=True)

    x = solve_mgs(A, b)

    print(f"[INFO] Вычисленное решение x: {x}", flush=True)

    residual = A @ x - b
    print(f"[INFO] Вектор невязки: {residual}", flush=True)

    assert np.allclose(A @ x, b, atol=1e-10)
    print("[OK] Проверка невязки пройдена: A @ x ~ b", flush=True)

    expected = np.array([1.0, 2.0, 3.0])
    print(f"[INFO] Ожидаемое решение: {expected}", flush=True)

    assert np.allclose(x, expected, atol=1e-10)
    print("[SUCCESS] test_orth_small_system успешно пройден", flush=True)


def test_orth_matches_numpy():
    """Сравнение с numpy.linalg.solve на случайной невырожденной матрице."""
    print("\n[TEST] test_orth_matches_numpy: сравнение solve_mgs с numpy.linalg.solve", flush=True)

    np.random.seed(42)
    n = 5

    print("[INFO] Используется фиксированное зерно генератора: 42", flush=True)
    print(f"[INFO] Размер случайной матрицы: {n}x{n}", flush=True)

    A = np.random.rand(n, n)
    A += np.eye(n) * 5.0
    b = np.random.rand(n)

    print(f"[INFO] Сгенерирована матрица A:\n{A}", flush=True)
    print(f"[INFO] Сгенерирован вектор b: {b}", flush=True)

    x = solve_mgs(A, b)
    expected = np.linalg.solve(A, b)

    print(f"[INFO] Решение solve_mgs: {x}", flush=True)
    print(f"[INFO] Решение numpy.linalg.solve: {expected}", flush=True)

    residual = A @ x - b
    print(f"[INFO] Вектор невязки для solve_mgs: {residual}", flush=True)

    assert np.allclose(A @ x, b, atol=1e-10)
    print("[OK] Проверка невязки пройдена: A @ x ~ b", flush=True)

    assert np.allclose(x, expected, atol=1e-10)
    print("[SUCCESS] test_orth_matches_numpy успешно пройден", flush=True)


def test_orth_singular_matrix():
    """Проверка реакции на вырожденную матрицу."""
    print("\n[TEST] test_orth_singular_matrix: проверка обработки вырожденной матрицы", flush=True)

    A = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 5.0, 7.0]
    ])
    b = np.array([1.0, 2.0, 3.0])

    print(f"[INFO] Вырожденная матрица A:\n{A}", flush=True)
    print(f"[INFO] Вектор b: {b}", flush=True)
    print("[INFO] Ожидается исключение numpy.linalg.LinAlgError", flush=True)

    with pytest.raises(np.linalg.LinAlgError):
        solve_mgs(A, b)

    print("[SUCCESS] LinAlgError корректно выброшено для вырожденной матрицы", flush=True)


def test_orth_non_square():
    """Проверка на неквадратную матрицу."""
    print("\n[TEST] test_orth_non_square: проверка обработки неквадратной матрицы A", flush=True)

    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    b = np.array([1.0, 2.0, 3.0])

    print(f"[INFO] Неквадратная матрица A формы {A.shape}:\n{A}", flush=True)
    print(f"[INFO] Вектор b: {b}", flush=True)
    print("[INFO] Ожидается исключение ValueError", flush=True)

    with pytest.raises(ValueError):
        solve_mgs(A, b)

    print("[SUCCESS] ValueError корректно выброшено для неквадратной матрицы A", flush=True)


def test_orth_invalid_b_size():
    """Проверка несовпадения размерности b."""
    print("\n[TEST] test_orth_invalid_b_size: проверка несовпадения размерности вектора b", flush=True)

    A = np.array([
        [2.0, 1.0],
        [1.0, 3.0]
    ])
    b = np.array([1.0, 2.0, 3.0])

    print(f"[INFO] Матрица A формы {A.shape}:\n{A}", flush=True)
    print(f"[INFO] Некорректный вектор b длины {len(b)}: {b}", flush=True)
    print("[INFO] Ожидается исключение ValueError", flush=True)

    with pytest.raises(ValueError):
        solve_mgs(A, b)

    print("[SUCCESS] ValueError корректно выброшено при несовпадении размерности b", flush=True)