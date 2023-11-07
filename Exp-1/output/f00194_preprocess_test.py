from f00194_preprocess import *
def test_preprocess():
    assert preprocess('Hello, world!') == [101, 7592, 1010, 2088, 999, 102]
    assert preprocess('This is a test.') == [101, 2023, 2003, 1037, 3231, 1012, 102]
    assert preprocess('I love NLP.') == [101, 1045, 2293, 2361, 1012, 102]
    assert preprocess('Transformers are great.') == [101, 16904, 2024, 2307, 1012, 102]
    assert preprocess('DistilBERT is awesome.') == [101, 12363, 14968, 2003, 7470, 1012, 102]

if __name__ == '__main__':
    test_preprocess()

