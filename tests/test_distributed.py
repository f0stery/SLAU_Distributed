from master import split_rows


def test_split_rows_even():
    """Равномерное распределение строк."""
    m = 12
    num_parts = 3

    ranges = split_rows(m, num_parts)

    assert len(ranges) == 3
    total_rows = sum(end - start for start, end in ranges)
    assert total_rows == m

    # Проверяем непрерывность покрытия
    assert ranges[0][0] == 0
    assert ranges[-1][1] == m
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i + 1][0]


def test_split_rows_uneven():
    """Неравномерное распределение строк."""
    m = 10
    num_parts = 3

    ranges = split_rows(m, num_parts)

    assert len(ranges) == 3
    total_rows = sum(end - start for start, end in ranges)
    assert total_rows == m

    # Разница между размерами блоков не должна быть больше 1
    sizes = [end - start for start, end in ranges]
    assert max(sizes) - min(sizes) <= 1