import os
import tempfile

from io_utils import load_matrix, load_vector, load_nodes


def test_load_matrix():
    """Тест загрузки матрицы из файла."""
    print("\n[TEST] test_load_matrix: проверка загрузки матрицы из текстового файла", flush=True)

    content = """3
4.0 1.0 1.0
1.0 5.0 1.0
1.0 1.0 6.0
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_file = f.name

    print(f"[INFO] Временный файл матрицы создан: {temp_file}", flush=True)

    try:
        A = load_matrix(temp_file)

        print(f"[INFO] Загруженная матрица: {A}", flush=True)
        print(f"[INFO] Размер матрицы: {len(A)} x {len(A[0])}", flush=True)

        assert len(A) == 3
        print("[OK] Количество строк матрицы равно 3", flush=True)

        assert len(A[0]) == 3
        print("[OK] Количество столбцов матрицы равно 3", flush=True)

        assert abs(A[0][0] - 4.0) < 1e-10
        print("[OK] Элемент A[0][0] корректен", flush=True)

        assert abs(A[2][2] - 6.0) < 1e-10
        print("[OK] Элемент A[2][2] корректен", flush=True)

        print("[SUCCESS] test_load_matrix успешно пройден", flush=True)

    finally:
        os.unlink(temp_file)
        print(f"[INFO] Временный файл удалён: {temp_file}", flush=True)


def test_load_vector():
    """Тест загрузки вектора из файла."""
    print("\n[TEST] test_load_vector: проверка загрузки вектора из текстового файла", flush=True)

    content = """3
9.0
14.0
21.0
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_file = f.name

    print(f"[INFO] Временный файл вектора создан: {temp_file}", flush=True)

    try:
        b = load_vector(temp_file)

        print(f"[INFO] Загруженный вектор: {b}", flush=True)
        print(f"[INFO] Размер вектора: {len(b)}", flush=True)

        assert len(b) == 3
        print("[OK] Размер вектора равен 3", flush=True)

        assert abs(b[0] - 9.0) < 1e-10
        print("[OK] Элемент b[0] корректен", flush=True)

        assert abs(b[2] - 21.0) < 1e-10
        print("[OK] Элемент b[2] корректен", flush=True)

        print("[SUCCESS] test_load_vector успешно пройден", flush=True)

    finally:
        os.unlink(temp_file)
        print(f"[INFO] Временный файл удалён: {temp_file}", flush=True)


def test_load_nodes():
    """Тест загрузки списка вычислительных узлов."""
    print("\n[TEST] test_load_nodes: проверка загрузки списка вычислительных узлов", flush=True)

    content = """127.0.0.1:5001
127.0.0.1:5002
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_file = f.name

    print(f"[INFO] Временный файл узлов создан: {temp_file}", flush=True)

    try:
        nodes = load_nodes(temp_file)

        print(f"[INFO] Загруженные узлы: {nodes}", flush=True)
        print(f"[INFO] Количество узлов: {len(nodes)}", flush=True)

        assert len(nodes) == 2
        print("[OK] Количество узлов равно 2", flush=True)

        assert nodes[0][0] == "127.0.0.1"
        print("[OK] IP первого узла корректен", flush=True)

        assert nodes[0][1] == 5001
        print("[OK] Порт первого узла корректен", flush=True)

        assert nodes[1][0] == "127.0.0.1"
        print("[OK] IP второго узла корректен", flush=True)

        assert nodes[1][1] == 5002
        print("[OK] Порт второго узла корректен", flush=True)

        print("[SUCCESS] test_load_nodes успешно пройден", flush=True)

    finally:
        os.unlink(temp_file)
        print(f"[INFO] Временный файл удалён: {temp_file}", flush=True)