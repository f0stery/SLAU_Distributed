import argparse
import time
from typing import List, Tuple

import numpy as np

from gauss_solver import solve_gauss
from master import distributed_mgs_stateful
from orth_solver import solve_mgs


def generate_well_conditioned_system(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M.T @ M + n * np.eye(n)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    return A, b, x_true


def run_benchmark(
    sizes: List[int],
    nodes: List[Tuple[str, int]],
    dist_limit: int,
    use_numpy_gauss_large: bool = True
):
    for n in sizes:
        print("\n" + "=" * 60)
        print(f"ТЕСТ: размерность {n}")
        print("=" * 60)

        A, b, x_true = generate_well_conditioned_system(n, seed=42 + n)

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
        x_mgs_dist = None
        t_mgs_dist = None

        if nodes and n <= dist_limit:
            t0 = time.perf_counter()
            x_mgs_dist = distributed_mgs_stateful(A, b, nodes, verbose=False)
            t_mgs_dist = time.perf_counter() - t0

        # Невязки
        r_gauss = np.linalg.norm(A @ x_gauss - b)
        r_mgs_seq = np.linalg.norm(A @ x_mgs_seq - b)

        # Ошибка относительно x_true
        e_gauss = np.linalg.norm(x_gauss - x_true)
        e_mgs_seq = np.linalg.norm(x_mgs_seq - x_true)

        print(f"{gauss_label}: {t_gauss:.6f} c")
        print(f"MGS   (последовательный): {t_mgs_seq:.6f} c")

        if x_mgs_dist is not None:
            r_mgs_dist = np.linalg.norm(A @ x_mgs_dist - b)
            e_mgs_dist = np.linalg.norm(x_mgs_dist - x_true)

            print(f"MGS   (распределённый):   {t_mgs_dist:.6f} c")

            print("\nНевязки:")
            print(f"||A*x_gauss - b||    = {r_gauss:.6e}")
            print(f"||A*x_mgs_seq - b||  = {r_mgs_seq:.6e}")
            print(f"||A*x_mgs_dist - b|| = {r_mgs_dist:.6e}")

            print("\nПогрешность относительно x_true:")
            print(f"||x_gauss - x_true||    = {e_gauss:.6e}")
            print(f"||x_mgs_seq - x_true||  = {e_mgs_seq:.6e}")
            print(f"||x_mgs_dist - x_true|| = {e_mgs_dist:.6e}")

            print("\nУскорение относительно распределённого MGS:")
            print(f"S_seqMGS = {t_mgs_seq / t_mgs_dist:.6f}")
            print(f"S_gauss  = {t_gauss / t_mgs_dist:.6f}")
        else:
            print("MGS   (распределённый):   ПРОПУЩЕН (n > dist_limit или нет узлов)")

            print("\nНевязки:")
            print(f"||A*x_gauss - b||    = {r_gauss:.6e}")
            print(f"||A*x_mgs_seq - b||  = {r_mgs_seq:.6e}")

            print("\nПогрешность относительно x_true:")
            print(f"||x_gauss - x_true||    = {e_gauss:.6e}")
            print(f"||x_mgs_seq - x_true||  = {e_mgs_seq:.6e}")


def parse_nodes_file(path: str):
    nodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            host, port = line.split(":")
            nodes.append((host.strip(), int(port.strip())))
    return nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Нагрузочный тест для решения СЛАУ")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10, 30, 50, 100, 300, 1000, 2000, 5000],
        help="Список размерностей"
    )
    parser.add_argument(
        "--nodes-file",
        default="data/nodes.txt",
        help="Файл списка worker-узлов"
    )
    parser.add_argument(
        "--dist-limit",
        type=int,
        default=300,
        help="Максимальная размерность для запуска TCP-распределённого MGS"
    )
    parser.add_argument(
        "--no-numpy-gauss-large",
        action="store_true",
        help="Не использовать np.linalg.solve для больших n"
    )
    parser.add_argument(
        "--no-nodes",
        action="store_true",
        help="Не использовать worker-узлы вообще"
    )
    args = parser.parse_args()

    nodes = []
    if not args.no_nodes:
        nodes = parse_nodes_file(args.nodes_file)

    run_benchmark(
        sizes=args.sizes,
        nodes=nodes,
        dist_limit=args.dist_limit,
        use_numpy_gauss_large=not args.no_numpy_gauss_large
    )