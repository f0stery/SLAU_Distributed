import time
import socket
import numpy as np

from gauss_solver import solve_gauss
from orth_solver import solve_mgs
from master import distributed_mgs


def generate_system(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Диагонально доминирующая матрица
    A = rng.random((n, n)) * 10.0
    A += np.eye(n) * (n + 10)

    x_true = rng.random(n) * 10.0
    b = A @ x_true

    return A, b, x_true


def residual_norm(A, x, b):
    return np.linalg.norm(A @ x - b)


def benchmark(n: int, nodes):
    print(f"\n{'=' * 60}")
    print(f"ТЕСТ: размерность {n}")
    print(f"{'=' * 60}")

    A, b, x_true = generate_system(n)

    # 1. Гаусс
    t0 = time.perf_counter()
    x_gauss = solve_gauss(A, b)
    t1 = time.perf_counter()

    # 2. Последовательный MGS
    t2 = time.perf_counter()
    x_mgs_seq = solve_mgs(A, b)
    t3 = time.perf_counter()

    # 3. Распределённый MGS
    t4 = time.perf_counter()
    x_mgs_dist = distributed_mgs(A, b, nodes)
    t5 = time.perf_counter()

    tg = t1 - t0
    tmgs = t3 - t2
    tdist = t5 - t4

    print(f"Гаусс (последовательный): {tg:.6f} c")
    print(f"MGS   (последовательный): {tmgs:.6f} c")
    print(f"MGS   (распределённый):   {tdist:.6f} c")

    print("\nНевязки:")
    print(f"||A*x_gauss - b||    = {residual_norm(A, x_gauss, b):.6e}")
    print(f"||A*x_mgs_seq - b||  = {residual_norm(A, x_mgs_seq, b):.6e}")
    print(f"||A*x_mgs_dist - b|| = {residual_norm(A, x_mgs_dist, b):.6e}")

    print("\nПогрешность относительно x_true:")
    print(f"||x_gauss - x_true||    = {np.linalg.norm(x_gauss - x_true):.6e}")
    print(f"||x_mgs_seq - x_true||  = {np.linalg.norm(x_mgs_seq - x_true):.6e}")
    print(f"||x_mgs_dist - x_true|| = {np.linalg.norm(x_mgs_dist - x_true):.6e}")

    if tdist > 0:
        print(f"\nУскорение относительно распределённого MGS:")
        print(f"S_seqMGS = {tmgs / tdist:.6f}")
        print(f"S_gauss  = {tg / tdist:.6f}")


if __name__ == "__main__":
    nodes = [
        ("127.0.0.1", 5001),
        ("127.0.0.1", 5002),
    ]

    for n in [10, 30, 50]:
        benchmark(n, nodes)