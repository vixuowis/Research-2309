from f00174_fibonacci import *
import unittest


class TestFibonacci(unittest.TestCase):

    def test_fibonacci(self):
        self.assertEqual(fibonacci(0), [0])
        self.assertEqual(fibonacci(1), [0, 1])
        self.assertEqual(fibonacci(2), [0, 1, 1])
        self.assertEqual(fibonacci(5), [0, 1, 1, 2, 3])
        self.assertEqual(fibonacci(10), [0, 1, 1, 2, 3, 5, 8])


if __name__ == '__main__':
    unittest.main()
