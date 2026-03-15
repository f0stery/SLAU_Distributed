import argparse
import pickle
import socket
import struct
import time
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
# Distributed MGS (simplified logging)
# ---------------------------

def distributed_mgs_stateful(A: np.ndarray, b: np.ndarray, nodes: List[Tuple[str, int]], verbose: bool = True) -> np.ndarray:
    """
    Распределённый MGS с параллельным опросом worker'ов.
    """
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

    print(f"\n[MGS] Инициализация {p} workers...")
    for idx, ((host, port), (rs, re)) in enumerate(zip(nodes, ranges)):
        print(f"  Worker {idx+1}: {host}:{port}, строки {rs}-{re}")
        payload = {
            "cmd": "init_data",
            "A_chunk": A[rs:re, :],
            "b_chunk": b[rs:re],
            "row_start": rs,
            "row_end": re,
        }
        send_task(host, port, payload, timeout=300.0)
    print(f"[MGS] Инициализация завершена")

    R = np.zeros((n, n), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    print(f"\n[MGS] Запуск основного цикла (n={n})...")
    start_time = time.perf_counter()

    # Создаем пул потоков для параллельных запросов
    with ThreadPoolExecutor(max_workers=p) as executor:
        for k in range(n):
            if verbose and (n <= 20 or (k+1) % 100 == 0 or k == 0):
                print(f"  Шаг {k + 1}/{n}")

            # ===== 1. ПАРАЛЛЕЛЬНЫЙ СБОР СТОЛБЦА И НОРМЫ =====
            col_global = np.zeros(m, dtype=np.float64)
            norm_sq = 0.0
            results = [None] * p

            def get_col_task(idx, host, port, rs, re, k):
                try:
                    resp = send_task(host, port, {"cmd": "get_col_norm", "k": k}, timeout=300.0)
                    return idx, rs, re, resp
                except Exception as e:
                    print(f"  Ошибка worker {host}:{port}: {e}")
                    return idx, rs, re, None

            futures = []
            for idx, ((host, port), (rs, re)) in enumerate(zip(nodes, ranges)):
                future = executor.submit(get_col_task, idx, host, port, rs, re, k)
                futures.append(future)

            for future in as_completed(futures):
                idx, rs, re, resp = future.result()
                if resp is not None:
                    col_local = np.asarray(resp["col_local"], dtype=np.float64)
                    norm_sq += float(resp["norm_sq_local"])
                    col_global[rs:re] = col_local
                    results[idx] = True

            if not all(results):
                raise RuntimeError("Не все worker'ы вернули результат")

            norm = np.sqrt(norm_sq)
            if norm < 1e-15:
                raise np.linalg.LinAlgError(f"Матрица вырождена на шаге k={k}")

            R[k, k] = norm
            q_global = col_global / norm

            # ===== 2. ПАРАЛЛЕЛЬНОЕ ВЫЧИСЛЕНИЕ y[k] =====
            yk = 0.0
            results = [None] * p

            def compute_yk_task(idx, host, port, rs, re, q_global):
                try:
                    q_local = q_global[rs:re]
                    resp = send_task(host, port, {"cmd": "compute_yk", "q_local": q_local}, timeout=300.0)
                    return idx, resp
                except Exception as e:
                    print(f"  Ошибка worker {host}:{port}: {e}")
                    return idx, None

            futures = []
            for idx, ((host, port), (rs, re)) in enumerate(zip(nodes, ranges)):
                future = executor.submit(compute_yk_task, idx, host, port, rs, re, q_global)
                futures.append(future)

            for future in as_completed(futures):
                idx, resp = future.result()
                if resp is not None:
                    yk += float(resp["yk_local"])
                    results[idx] = True

            if not all(results):
                raise RuntimeError("Не все worker'ы вернули результат")
            y[k] = yk

            # ===== 3. ПАРАЛЛЕЛЬНОЕ ВЫЧИСЛЕНИЕ СКАЛЯРНЫХ ПРОИЗВЕДЕНИЙ =====
            tail_len = n - (k + 1)
            if tail_len > 0:
                r_tail = np.zeros(tail_len, dtype=np.float64)
                results = [None] * p

                def compute_dots_task(idx, host, port, rs, re, q_global, k):
                    try:
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
                        return idx, resp
                    except Exception as e:
                        print(f"  Ошибка worker {host}:{port}: {e}")
                        return idx, None

                futures = []
                for idx, ((host, port), (rs, re)) in enumerate(zip(nodes, ranges)):
                    future = executor.submit(compute_dots_task, idx, host, port, rs, re, q_global, k)
                    futures.append(future)

                for future in as_completed(futures):
                    idx, resp = future.result()
                    if resp is not None:
                        dots_local = np.asarray(resp["dots_local"], dtype=np.float64)
                        r_tail += dots_local
                        results[idx] = True

                if not all(results):
                    raise RuntimeError("Не все worker'ы вернули результат")

                R[k, (k + 1):] = r_tail

                # ===== 4. ПАРАЛЛЕЛЬНОЕ ОБНОВЛЕНИЕ =====
                def update_task(host, port, rs, re, q_global, r_tail, k):
                    try:
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
                        return True
                    except Exception as e:
                        print(f"  Ошибка worker {host}:{port}: {e}")
                        return False

                futures = []
                for (host, port), (rs, re) in zip(nodes, ranges):
                    future = executor.submit(update_task, host, port, rs, re, q_global, r_tail, k)
                    futures.append(future)

                update_results = [f.result() for f in futures]
                if not all(update_results):
                    raise RuntimeError("Не все worker'ы выполнили обновление")

    total_time = time.perf_counter() - start_time
    print(f"\n[MGS] Основной цикл завершен за {total_time:.2f} с")

    # Решаем R x = y
    x = back_substitution(R, y)
    return x


# ---------------------------
# Main runner
# ---------------------------

def run_from_files(matrix_path: str, vector_path: str, nodes_path: str):
    print("\n" + "=" * 60)
    print("ЗАПУСК MASTER.PY")
    print("=" * 60)

    print("\nЗагрузка данных...")
    A = load_matrix(matrix_path)
    b = load_vector(vector_path)
    nodes = load_nodes(nodes_path)

    n = A.shape[0]
    print(f"  Размерность системы: {n}")
    print(f"  Узлов: {len(nodes)}")

    # Гаусс
    print("\nРешение методом Гаусса...")
    t0 = time.perf_counter()
    x_gauss = solve_gauss(A, b)
    t_gauss = time.perf_counter() - t0
    print(f"  Время: {t_gauss:.3f} с")

    # Последовательный MGS
    print("\nРешение последовательным MGS...")
    t0 = time.perf_counter()
    x_mgs_seq = solve_mgs(A, b)
    t_mgs_seq = time.perf_counter() - t0
    print(f"  Время: {t_mgs_seq:.3f} с")

    # Распределённый MGS
    print("\nРешение распределённым MGS...")
    t0 = time.perf_counter()
    x_mgs_dist = distributed_mgs_stateful(A, b, nodes, verbose=True)
    t_mgs_dist = time.perf_counter() - t0
    print(f"  Время: {t_mgs_dist:.3f} с")

    # Невязки
    r_gauss = np.linalg.norm(A @ x_gauss - b)
    r_mgs_seq = np.linalg.norm(A @ x_mgs_seq - b)
    r_mgs_dist = np.linalg.norm(A @ x_mgs_dist - b)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)

    print("\nВремя выполнения:")
    print(f"  Гаусс (последовательный): {t_gauss:.6f} с")
    print(f"  MGS (последовательный): {t_mgs_seq:.6f} с")
    print(f"  MGS (распределённый): {t_mgs_dist:.6f} с")

    if t_mgs_dist > 0:
        print(f"\nУскорение (посл. MGS / распред. MGS): {t_mgs_seq / t_mgs_dist:.6f}x")
        print(f"Ускорение (Гаусс / распред. MGS): {t_gauss / t_mgs_dist:.6f}x")

    print("\nНевязки ||Ax-b||:")
    print(f"  Гаусс: {r_gauss:.6e}")
    print(f"  MGS последовательный: {r_mgs_seq:.6e}")
    print(f"  MGS распределённый: {r_mgs_dist:.6e}")

    print("\n" + "=" * 60)
    print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master для распределённого решения СЛАУ методом ортогонализации")
    parser.add_argument("--matrix", default="data/matrix.txt", help="Файл матрицы")
    parser.add_argument("--vector", default="data/vector.txt", help="Файл вектора b")
    parser.add_argument("--nodes", default="data/nodes.txt", help="Файл списка worker-узлов")
    args = parser.parse_args()

    run_from_files(
        args.matrix,
        args.vector,
        args.nodes
    )