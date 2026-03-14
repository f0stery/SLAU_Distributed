import argparse
import pickle
import socket
import struct
import time
from typing import List, Tuple

import numpy as np

from gauss_solver import solve_gauss
from io_utils import load_matrix, load_nodes, load_vector
from orth_solver import solve_mgs


# ---------------------------
# Framed TCP helpers
# ---------------------------

def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Соединение закрыто во время чтения данных")
        data.extend(chunk)
    return bytes(data)


def recv_message(sock: socket.socket):
    header = recv_exact(sock, 8)
    (length,) = struct.unpack("!Q", header)
    payload = recv_exact(sock, length)
    return pickle.loads(payload)


def send_message(sock: socket.socket, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    sock.sendall(header + payload)


def send_task(host: str, port: int, payload: dict, timeout: float = 120.0) -> dict:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        send_message(sock, payload)
        resp = recv_message(sock)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Worker {host}:{port} error: {resp['error']}")
        return resp


# ---------------------------
# Utils
# ---------------------------

def split_rows(m: int, num_parts: int) -> List[Tuple[int, int]]:
    parts = []
    base = m // num_parts
    rem = m % num_parts
    start = 0
    for i in range(num_parts):
        size = base + (1 if i < rem else 0)
        end = start + size
        parts.append((start, end))
        start = end
    return parts


def back_substitution(R: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = R.shape[0]
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = np.dot(R[i, i + 1:], x[i + 1:])
        x[i] = (y[i] - s) / R[i, i]
    return x


# ---------------------------
# Distributed MGS (stateful workers)
# ---------------------------

def distributed_mgs_stateful(A: np.ndarray, b: np.ndarray, nodes: List[Tuple[str, int]], verbose: bool = True) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = A.shape
    if m != n:
        raise ValueError("Для данной реализации требуется квадратная матрица A (n x n)")
    if b.shape[0] != n:
        raise ValueError("Размер b должен совпадать с размерностью системы")
    if not nodes:
        raise ValueError("Список worker-узлов пуст")

    p = len(nodes)
    ranges = split_rows(m, p)

    # 1) INIT: загружаем данные на workers один раз
    for (host, port), (rs, re) in zip(nodes, ranges):
        payload = {
            "cmd": "init_data",
            "A_chunk": A[rs:re, :],
            "b_chunk": b[rs:re],
            "row_start": rs,
            "row_end": re,
        }
        send_task(host, port, payload, timeout=300.0)

    R = np.zeros((n, n), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    # 2) Основной цикл MGS
    for k in range(n):
        if verbose and (n <= 20 or k % max(1, n // 20) == 0):
            print(f"[MGS] шаг {k + 1}/{n}")

        # 2.1) Собрать k-й столбец и норму
        col_global = np.zeros(m, dtype=np.float64)
        norm_sq = 0.0

        for (host, port), (rs, re) in zip(nodes, ranges):
            resp = send_task(host, port, {"cmd": "get_col_norm", "k": k}, timeout=300.0)
            col_local = np.asarray(resp["col_local"], dtype=np.float64)
            norm_sq += float(resp["norm_sq_local"])
            col_global[rs:re] = col_local

        norm = np.sqrt(norm_sq)
        if norm < 1e-15:
            raise np.linalg.LinAlgError(f"Матрица вырождена или почти вырождена на шаге k={k}")

        R[k, k] = norm
        q_global = col_global / norm

        # 2.2) y[k] = q_k^T b
        yk = 0.0
        for (host, port), (rs, re) in zip(nodes, ranges):
            q_local = q_global[rs:re]
            resp = send_task(host, port, {"cmd": "compute_yk", "q_local": q_local}, timeout=300.0)
            yk += float(resp["yk_local"])
        y[k] = yk

        # 2.3) r[k, k+1:] = q_k^T A[:, k+1:]
        tail_len = n - (k + 1)
        if tail_len > 0:
            r_tail = np.zeros(tail_len, dtype=np.float64)

            for (host, port), (rs, re) in zip(nodes, ranges):
                q_local = q_global[rs:re]
                resp = send_task(
                    host,
                    port,
                    {
                        "cmd": "compute_dots",
                        "k": k,
                        "q_local": q_local,
                    },
                    timeout=300.0
                )
                dots_local = np.asarray(resp["dots_local"], dtype=np.float64)
                r_tail += dots_local

            R[k, (k + 1):] = r_tail

            # 2.4) A[:, k+1:] -= q_k * r_tail
            for (host, port), (rs, re) in zip(nodes, ranges):
                q_local = q_global[rs:re]
                send_task(
                    host,
                    port,
                    {
                        "cmd": "update_tail",
                        "k": k,
                        "q_local": q_local,
                        "r_tail": r_tail,
                    },
                    timeout=300.0
                )

    # 3) Решаем R x = y
    x = back_substitution(R, y)
    return x


# ---------------------------
# Main runner
# ---------------------------

def run_from_files(matrix_path: str, vector_path: str, nodes_path: str, use_numpy_gauss_large: bool = True):
    A = load_matrix(matrix_path)
    b = load_vector(vector_path)
    nodes = load_nodes(nodes_path)

    n = A.shape[0]
    print(f"Размерность системы: {n}")
    print(f"Узлов: {len(nodes)}")

    # Гаусс / эталон
    t0 = time.perf_counter()
    if use_numpy_gauss_large and n >= 1000:
        x_gauss = np.linalg.solve(A, b)
        gauss_label = "NumPy solve (LAPACK)"
    else:
        x_gauss = solve_gauss(A, b)
        gauss_label = "Гаусс (последовательный)"
    t_gauss = time.perf_counter() - t0

    # Последовательный MGS
    t0 = time.perf_counter()
    x_mgs_seq = solve_mgs(A, b)
    t_mgs_seq = time.perf_counter() - t0

    # Распределённый MGS
    t0 = time.perf_counter()
    x_mgs_dist = distributed_mgs_stateful(A, b, nodes, verbose=True)
    t_mgs_dist = time.perf_counter() - t0

    # Невязки
    r_gauss = np.linalg.norm(A @ x_gauss - b)
    r_mgs_seq = np.linalg.norm(A @ x_mgs_seq - b)
    r_mgs_dist = np.linalg.norm(A @ x_mgs_dist - b)

    # Различие решений
    d_seq = np.linalg.norm(x_gauss - x_mgs_seq)
    d_dist = np.linalg.norm(x_gauss - x_mgs_dist)

    print("\n=== Время выполнения ===")
    print(f"{gauss_label}: {t_gauss:.6f} c")
    print(f"MGS   (последовательный): {t_mgs_seq:.6f} c")
    print(f"MGS   (распределённый):   {t_mgs_dist:.6f} c")

    print("\n=== Невязки ===")
    print(f"||A*x_gauss - b||    = {r_gauss:.6e}")
    print(f"||A*x_mgs_seq - b||  = {r_mgs_seq:.6e}")
    print(f"||A*x_mgs_dist - b|| = {r_mgs_dist:.6e}")

    print("\n=== Различие решений ===")
    print(f"||x_gauss - x_mgs_seq||  = {d_seq:.6e}")
    print(f"||x_gauss - x_mgs_dist|| = {d_dist:.6e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master для распределённого решения СЛАУ методом ортогонализации")
    parser.add_argument("--matrix", default="data/matrix.txt", help="Файл матрицы")
    parser.add_argument("--vector", default="data/vector.txt", help="Файл вектора b")
    parser.add_argument("--nodes", default="data/nodes.txt", help="Файл списка worker-узлов")
    parser.add_argument("--no-numpy-gauss-large", action="store_true", help="Не использовать np.linalg.solve для больших n")
    args = parser.parse_args()

    run_from_files(
        args.matrix,
        args.vector,
        args.nodes,
        use_numpy_gauss_large=not args.no_numpy_gauss_large
    )