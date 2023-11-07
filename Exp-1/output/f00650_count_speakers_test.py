from f00650_count_speakers import *
import pandas as pd
import unittest


def test_count_speakers(unittest.TestCase):
    def test_count_speakers(self):
        dataset = pd.DataFrame({'speaker_id': [1, 2, 3, 1, 2, 3]})
        self.assertEqual(count_speakers(dataset), 3)

    def test_count_speakers_empty_dataset(self):
        dataset = pd.DataFrame()
        self.assertEqual(count_speakers(dataset), 0)


if __name__ == '__main__':
    unittest.main()
