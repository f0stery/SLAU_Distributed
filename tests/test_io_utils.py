import os
import tempfile

from io_utils import load_matrix, load_vector, load_nodes


def test_load_matrix():
    """Тест загрузки матрицы из файла."""
    content = """3
4.0 1.0 1.0
1.0 5.0 1.0
1.0 1.0 6.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_file = f.name

    try:
        A = load_matrix(temp_file)
        assert len(A) == 3
        assert len(A[0]) == 3
        assert abs(A[0][0] - 4.0) < 1e-10
        assert abs(A[2][2] - 6.0) < 1e-10
    finally:
        os.unlink(temp_file)


def test_load_vector():
    """Тест загрузки вектора из файла."""
    content = """3
9.0
14.0
21.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_file = f.name

    try:
        b = load_vector(temp_file)
        assert len(b) == 3
        assert abs(b[0] - 9.0) < 1e-10
        assert abs(b[2] - 21.0) < 1e-10
    finally:
        os.unlink(temp_file)


def test_load_nodes():
    """Тест загрузки списка вычислительных узлов."""
    content = """127.0.0.1:5001
127.0.0.1:5002
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_file = f.name

    try:
        nodes = load_nodes(temp_file)
        assert len(nodes) == 2
        assert nodes[0][0] == "127.0.0.1"
        assert nodes[0][1] == 5001
        assert nodes[1][0] == "127.0.0.1"
        assert nodes[1][1] == 5002
    finally:
        os.unlink(temp_file)