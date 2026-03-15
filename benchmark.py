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
    total_tests = len(sizes)

    print("\n" + "=" * 60)
    print("ЗАПУСК НАГРУЗОЧНОГО ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"Всего тестов: {total_tests}")
    print(f"Worker'ы: {len(nodes)}")
    print(f"dist-limit: {dist_limit}")
    print("=" * 60)

    results = []

    for test_idx, n in enumerate(sizes, 1):
        print("\n" + "=" * 60)
        print(f"ТЕСТ {test_idx}/{total_tests}: размерность {n}")
        print("=" * 60)

        # Генерация системы
        print(f"\nГенерация системы размерности {n}...")
        gen_start = time.perf_counter()
        A, b, x_true = generate_well_conditioned_system(n, seed=42 + n)
        gen_time = time.perf_counter() - gen_start
        print(f"  Время: {gen_time:.3f} с")
        print(f"  Число обусловленности: {np.linalg.cond(A):.2e}")

        # Гаусс
        print(f"\nРешение методом Гаусса...")
        t0 = time.perf_counter()
        x_gauss = solve_gauss(A, b)
        gauss_label = "Гаусс (последовательный)"
        t_gauss = time.perf_counter() - t0
        print(f"  Время: {t_gauss:.3f} с")

        # Последовательный MGS
        print(f"\nРешение последовательным MGS...")
        t0 = time.perf_counter()
        x_mgs_seq = solve_mgs(A, b)
        t_mgs_seq = time.perf_counter() - t0
        print(f"  Время: {t_mgs_seq:.3f} с")

        # Распределённый MGS
        x_mgs_dist = None
        t_mgs_dist = None
        speedup_seq = None
        speedup_gauss = None

        if nodes and n <= dist_limit:
            print(f"\nРешение распределённым MGS...")
            print(f"  Worker'ы: {len(nodes)}")

            t0 = time.perf_counter()
            x_mgs_dist = distributed_mgs_stateful(A, b, nodes, verbose=True)
            t_mgs_dist = time.perf_counter() - t0

            speedup_seq = t_mgs_seq / t_mgs_dist if t_mgs_dist > 0 else 0
            speedup_gauss = t_gauss / t_mgs_dist if t_mgs_dist > 0 else 0

            print(f"  Время: {t_mgs_dist:.3f} с")
            print(f"  Ускорение (посл. MGS / распред. MGS): {speedup_seq:.2f}x")
            print(f"  Ускорение (Гаусс / распред. MGS): {speedup_gauss:.2f}x")
        else:
            if n > dist_limit:
                print(f"\nРаспределённый MGS пропущен: n={n} > dist_limit={dist_limit}")
            else:
                print(f"\nРаспределённый MGS пропущен: нет узлов")

        # Вычисление метрик
        r_gauss = np.linalg.norm(A @ x_gauss - b)
        r_mgs_seq = np.linalg.norm(A @ x_mgs_seq - b)

        e_gauss = np.linalg.norm(x_gauss - x_true)
        e_mgs_seq = np.linalg.norm(x_mgs_seq - x_true)

        test_result = {
            'n': n,
            't_gauss': t_gauss,
            't_mgs_seq': t_mgs_seq,
            't_mgs_dist': t_mgs_dist,
            'r_gauss': r_gauss,
            'r_mgs_seq': r_mgs_seq,
            'e_gauss': e_gauss,
            'e_mgs_seq': e_mgs_seq,
            'speedup_seq': speedup_seq if t_mgs_dist else None,
            'speedup_gauss': speedup_gauss if t_mgs_dist else None
        }

        if x_mgs_dist is not None:
            test_result['r_mgs_dist'] = np.linalg.norm(A @ x_mgs_dist - b)
            test_result['e_mgs_dist'] = np.linalg.norm(x_mgs_dist - x_true)

        results.append(test_result)

        # Вывод результатов
        print("\n" + "-" * 40)
        print("РЕЗУЛЬТАТЫ ТЕСТА:")
        print("-" * 40)

        print(f"\nВремя выполнения:")
        print(f"  {gauss_label}: {t_gauss:.6f} с")
        print(f"  MGS (последовательный): {t_mgs_seq:.6f} с")
        if t_mgs_dist:
            print(f"  MGS (распределённый): {t_mgs_dist:.6f} с")

        print(f"\nНевязки ||Ax-b||:")
        print(f"  Гаусс: {r_gauss:.6e}")
        print(f"  MGS последовательный: {r_mgs_seq:.6e}")
        if x_mgs_dist is not None:
            print(f"  MGS распределённый: {test_result['r_mgs_dist']:.6e}")

        print(f"\nПогрешность относительно x_true:")
        print(f"  Гаусс: {e_gauss:.6e}")
        print(f"  MGS последовательный: {e_mgs_seq:.6e}")
        if x_mgs_dist is not None:
            print(f"  MGS распределённый: {test_result['e_mgs_dist']:.6e}")

        print(f"\n{'-' * 40}")
        print(f"ТЕСТ {test_idx}/{total_tests} ЗАВЕРШЕН")
        print(f"{'-' * 40}")

    # Итоговые результаты
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)

    print("\n{:<8} {:<15} {:<15} {:<15} {:<12}".format(
        "n", "Гаусс (с)", "MGS посл (с)", "MGS распр (с)", "Ускорение"))
    print("-" * 65)

    for r in results:
        if r['t_mgs_dist']:
            print("{:<8} {:<15.6f} {:<15.6f} {:<15.6f} {:<12.2f}x".format(
                r['n'], r['t_gauss'], r['t_mgs_seq'], r['t_mgs_dist'], r['speedup_seq']))
        else:
            print("{:<8} {:<15.6f} {:<15.6f} {:<15} {:<12}".format(
                r['n'], r['t_gauss'], r['t_mgs_seq'], "N/A", "N/A"))

    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


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
    parser = argparse.ArgumentParser(description="Нагрузочное тестирование решения СЛАУ")
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
        default=3000,
        help="Максимальная размерность для распределённого MGS"
    )
    parser.add_argument(
        "--no-numpy-gauss-large",
        action="store_true",
        help="Не использовать np.linalg.solve для больших n"
    )
    parser.add_argument(
        "--no-nodes",
        action="store_true",
        help="Не использовать worker-узлы"
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