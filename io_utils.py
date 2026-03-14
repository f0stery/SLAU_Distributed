import numpy as np


def load_matrix(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    data = []
    for i in range(1, n + 1):
        row = list(map(float, lines[i].split()))
        if len(row) != n:
            raise ValueError(f"Строка {i} матрицы имеет неверное количество элементов")
        data.append(row)

    return np.array(data, dtype=np.float64)


def load_vector(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    data = [float(lines[i]) for i in range(1, n + 1)]
    return np.array(data, dtype=np.float64)


def load_nodes(path: str):
    nodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            host, port = line.split(":")
            nodes.append((host, int(port)))
    return nodes