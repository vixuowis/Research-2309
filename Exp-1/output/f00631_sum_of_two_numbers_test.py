from f00631_sum_of_two_numbers import *
def test_sum_of_two_numbers():
    assert sum_of_two_numbers(2, 3) == 5
    assert sum_of_two_numbers(-5, 10) == 5
    assert sum_of_two_numbers(0, 0) == 0
    assert sum_of_two_numbers(100, -100) == 0
    assert sum_of_two_numbers(50, 50) == 100


def main():
    test_sum_of_two_numbers()


if __name__ == '__main__':
    main()
