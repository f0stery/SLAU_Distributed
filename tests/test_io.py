import numpy as np
from pathlib import Path
from io_utils import load_matrix, load_vector, load_nodes


def test_io_loading(tmp_path: Path):
    matrix_file = tmp_path / "matrix.txt"
    vector_file = tmp_path / "vector.txt"
    nodes_file = tmp_path / "nodes.txt"

    matrix_file.write_text(
        "2\n"
        "1 2\n"
        "3 4\n",
        encoding="utf-8"
    )

    vector_file.write_text(
        "2\n"
        "5\n"
        "6\n",
        encoding="utf-8"
    )

    nodes_file.write_text(
        "127.0.0.1:5001\n"
        "127.0.0.1:5002\n",
        encoding="utf-8"
    )

    A = load_matrix(str(matrix_file))
    b = load_vector(str(vector_file))
    nodes = load_nodes(str(nodes_file))

    assert A.shape == (2, 2)
    assert np.allclose(A, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(b, np.array([5.0, 6.0]))
    assert nodes == [("127.0.0.1", 5001), ("127.0.0.1", 5002)]