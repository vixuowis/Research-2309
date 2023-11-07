from f00787_model_init import *
def test_model_init():
    assert isinstance(model_init(trial), AutoModelForSequenceClassification)

    # Add more test cases here

if __name__ == '__main__':
    test_model_init()
