from master import split_rows


def test_split_rows_even():
    """Равномерное распределение строк."""
    print("\n[TEST] test_split_rows_even: проверка равномерного распределения строк")

    m = 12
    num_parts = 3

    print(f"[INFO] Входные данные: m={m}, num_parts={num_parts}")

    ranges = split_rows(m, num_parts)

    print(f"[INFO] Полученные диапазоны: {ranges}")

    assert len(ranges) == 3
    print("[OK] Количество диапазонов корректно")

    total_rows = sum(end - start for start, end in ranges)
    print(f"[INFO] Суммарное число строк: {total_rows}")

    assert total_rows == m
    print("[OK] Суммарное число строк совпадает с m")

    assert ranges[0][0] == 0
    print("[OK] Первый диапазон начинается с 0")

    assert ranges[-1][1] == m
    print("[OK] Последний диапазон заканчивается на m")

    for i in range(len(ranges) - 1):
        print(f"[INFO] Проверка стыка диапазонов {i} и {i+1}: {ranges[i][1]} == {ranges[i + 1][0]}")
        assert ranges[i][1] == ranges[i + 1][0]

    print("[SUCCESS] test_split_rows_even успешно пройден")


def test_split_rows_uneven():
    """Неравномерное распределение строк."""
    print("\n[TEST] test_split_rows_uneven: проверка неравномерного распределения строк")

    m = 10
    num_parts = 3

    print(f"[INFO] Входные данные: m={m}, num_parts={num_parts}")

    ranges = split_rows(m, num_parts)

    print(f"[INFO] Полученные диапазоны: {ranges}")

    assert len(ranges) == 3
    print("[OK] Количество диапазонов корректно")

    total_rows = sum(end - start for start, end in ranges)
    print(f"[INFO] Суммарное число строк: {total_rows}")

    assert total_rows == m
    print("[OK] Суммарное число строк совпадает с m")

    sizes = [end - start for start, end in ranges]
    print(f"[INFO] Размеры блоков: {sizes}")

    assert max(sizes) - min(sizes) <= 1
    print("[OK] Разница между размерами блоков не превышает 1")

    print("[SUCCESS] test_split_rows_uneven успешно пройден")