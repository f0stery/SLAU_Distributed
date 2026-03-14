import argparse
import os

import numpy as np


def generate_well_conditioned_system(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M.T @ M + n * np.eye(n)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    return A, b, x_true


def save_matrix(path: str, A: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        n = A.shape[0]
        f.write(f"{n}\n")
        for row in A:
            f.write(" ".join(f"{x:.12g}" for x in row) + "\n")


def save_vector(path: str, b: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        n = b.shape[0]
        f.write(f"{n}\n")
        for x in b:
            f.write(f"{x:.12g}\n")


def save_nodes(path: str, nodes):
    with open(path, "w", encoding="utf-8") as f:
        for host, port in nodes:
            f.write(f"{host}:{port}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    A, b, _ = generate_well_conditioned_system(args.n, seed=args.seed)

    save_matrix(os.path.join(args.out_dir, "matrix.txt"), A)
    save_vector(os.path.join(args.out_dir, "vector.txt"), b)
    save_nodes(os.path.join(args.out_dir, "nodes.txt"), [("127.0.0.1", 5001), ("127.0.0.1", 5002)])

    print(f"Сгенерированы данные для n={args.n} в папке {args.out_dir}")