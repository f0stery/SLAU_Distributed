import argparse
import time
import socket
import pickle
import numpy as np

from io_utils import load_matrix, load_vector, load_nodes
from orth_solver import solve_mgs
from gauss_solver import solve_gauss


def send_task(host, port, payload):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        data = pickle.dumps(payload)
        s.sendall(len(data).to_bytes(8, "big"))
        s.sendall(data)

        size = int.from_bytes(s.recv(8), "big")
        chunks = []
        received = 0
        while received < size:
            chunk = s.recv(min(4096, size - received))
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)

        result = pickle.loads(b"".join(chunks))
        return result


def split_rows(A, start, end):
    return A[start:end, :]


def distributed_mgs(A, b, nodes):
    """
    Распределённый Modified Gram-Schmidt:
    - строки матрицы A делятся между worker-узлами
    - на каждой итерации worker вычисляет локальные части:
        * локальный вклад в норму столбца
        * локальные скалярные произведения
        * обновление своих строк
    """
    A_work = A.astype(float).copy()
    m, n = A_work.shape

    if m != n:
        raise ValueError("Для решения СЛАУ требуется квадратная матрица")

    b_work = b.astype(float).copy()
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    # Разбиение строк между узлами
    row_ranges = []
    chunk = (m + len(nodes) - 1) // len(nodes)
    start = 0
    for _ in nodes:
        end = min(start + chunk, m)
        row_ranges.append((start, end))
        start = end

    for k in range(n):
        # 1) Собираем k-й столбец по кускам и считаем глобальную норму
        local_cols = []
        norm_sq = 0.0

        for idx, (host, port) in enumerate(nodes):
            rs, re = row_ranges[idx]
            A_chunk = A_work[rs:re, :]
            b_chunk = b_work[rs:re]

            payload = {
                "cmd": "get_column_and_norm",
                "A_chunk": A_chunk,
                "b_chunk": b_chunk,
                "k": k,
            }
            result = send_task(host, port, payload)
            if "error" in result:
                raise RuntimeError(f"Worker {host}:{port} вернул ошибку: {result['error']}")
            local_col = result["col_k"]
            local_cols.append(local_col)
            norm_sq += result["norm_sq"]

        R[k, k] = np.sqrt(norm_sq)
        if abs(R[k, k]) < 1e-14:
            raise ValueError("Матрица вырождена или близка к вырожденной")

        # Формируем q_k
        qk = np.zeros(m, dtype=float)
        for idx, (rs, re) in enumerate(row_ranges):
            qk[rs:re] = local_cols[idx] / R[k, k]

        Q[:, k] = qk

        # 2) Для каждого j > k вычисляем R[k, j] = q_k^T a_j
        for j in range(k + 1, n):
            dot_sum = 0.0
            for idx, (host, port) in enumerate(nodes):
                rs, re = row_ranges[idx]
                A_chunk = A_work[rs:re, :]

                payload = {
                    "cmd": "dot_with_q",
                    "A_chunk": A_chunk,
                    "q_chunk": qk[rs:re],
                    "j": j,
                }
                result = send_task(host, port, payload)
                dot_sum += result["dot"]

            R[k, j] = dot_sum

            # 3) Обновляем столбец j: a_j = a_j - R[k,j] * q_k
            for idx, (host, port) in enumerate(nodes):
                rs, re = row_ranges[idx]
                A_chunk = A_work[rs:re, :]

                payload = {
                    "cmd": "update_column",
                    "A_chunk": A_chunk,
                    "q_chunk": qk[rs:re],
                    "j": j,
                    "rkj": R[k, j],
                }
                result = send_task(host, port, payload)
                A_work[rs:re, :] = result["A_chunk"]

    # Решаем R x = Q^T b
    y = Q.T @ b_work
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= R[i, j] * x[j]
        x[i] = s / R[i, i]

    return x


def run_from_files(matrix_path, vector_path, nodes_path):
    A = load_matrix(matrix_path)
    b = load_vector(vector_path)
    nodes = load_nodes(nodes_path)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    if A.shape[0] != len(b):
        raise ValueError("Размерность матрицы и вектора b не совпадает")
    if A.shape[0] > 5000:
        raise ValueError("Превышено ограничение: максимум 5000 неизвестных")

    print(f"Размерность системы: {A.shape[0]}")
    print(f"Узлов: {len(nodes)}")

    # Гаусс
    t0 = time.perf_counter()
    x_gauss = solve_gauss(A, b)
    t1 = time.perf_counter()

    # Последовательный MGS
    t2 = time.perf_counter()
    x_mgs_seq = solve_mgs(A, b)
    t3 = time.perf_counter()

    # Распределённый MGS
    t4 = time.perf_counter()
    x_mgs_dist = distributed_mgs(A, b, nodes)
    t5 = time.perf_counter()

    print("\n=== Время выполнения ===")
    print(f"Гаусс (последовательный): {t1 - t0:.6f} c")
    print(f"MGS   (последовательный): {t3 - t2:.6f} c")
    print(f"MGS   (распределённый):   {t5 - t4:.6f} c")

    print("\n=== Невязки ===")
    print(f"||A*x_gauss - b||    = {np.linalg.norm(A @ x_gauss - b):.6e}")
    print(f"||A*x_mgs_seq - b||  = {np.linalg.norm(A @ x_mgs_seq - b):.6e}")
    print(f"||A*x_mgs_dist - b|| = {np.linalg.norm(A @ x_mgs_dist - b):.6e}")

    print("\n=== Различие решений ===")
    print(f"||x_gauss - x_mgs_seq||  = {np.linalg.norm(x_gauss - x_mgs_seq):.6e}")
    print(f"||x_gauss - x_mgs_dist|| = {np.linalg.norm(x_gauss - x_mgs_dist):.6e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Распределённое решение СЛАУ методом ортогонализации")
    parser.add_argument("--matrix", default="data/matrix.txt", help="Путь к файлу матрицы")
    parser.add_argument("--vector", default="data/vector.txt", help="Путь к файлу вектора b")
    parser.add_argument("--nodes", default="data/nodes.txt", help="Путь к файлу списка узлов")

    args = parser.parse_args()
    run_from_files(args.matrix, args.vector, args.nodes)