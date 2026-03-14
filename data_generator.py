import numpy as np
from pathlib import Path


def generate_system(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Диагонально доминирующая матрица => устойчивая и невырожденная
    A = rng.random((n, n)) * 10.0
    A += np.eye(n) * (n + 10)

    x_true = rng.random(n) * 10.0
    b = A @ x_true

    return A, b, x_true


def save_matrix(path: str, A: np.ndarray):
    n = A.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for i in range(n):
            row = " ".join(f"{x:.10f}" for x in A[i])
            f.write(row + "\n")


def save_vector(path: str, b: np.ndarray):
    n = len(b)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        for x in b:
            f.write(f"{x:.10f}\n")


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    n = 50  # можешь менять: 10, 50, 100, 200, 500
    A, b, x_true = generate_system(n)

    save_matrix(data_dir / "matrix.txt", A)
    save_vector(data_dir / "vector.txt", b)

    print(f"Сгенерирована система размерности {n}")
    print("Файлы обновлены:")
    print(" - data/matrix.txt")
    print(" - data/vector.txt")


if __name__ == "__main__":
    main()