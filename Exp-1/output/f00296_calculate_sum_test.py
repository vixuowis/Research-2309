from f00296_calculate_sum import *
import unittest


class TestCalculateSum(unittest.TestCase):
    def test_calculate_sum(self):
        # Test case 1
        numbers1 = [1, 2, 3, 4, 5]
        expected1 = 15
        self.assertEqual(calculate_sum(numbers1), expected1)

        # Test case 2
        numbers2 = [-1, -2, -3, -4, -5]
        expected2 = -15
        self.assertEqual(calculate_sum(numbers2), expected2)

        # Test case 3
        numbers3 = []
        expected3 = 0
        self.assertEqual(calculate_sum(numbers3), expected3)


if __name__ == '__main__':
    unittest.main()
