import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from master import split_rows


class TestDistributed:
    """Минимальные тесты для распределенной части"""

    def test_split_rows(self):
        """Тест функции распределения строк"""
        m = 10
        num_parts = 3

        ranges = split_rows(m, num_parts)

        assert len(ranges) == 3
        # Проверяем, что покрыты все строки
        total_rows = sum(end - start for start, end in ranges)
        assert total_rows == m