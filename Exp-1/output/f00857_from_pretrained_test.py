from f00857_from_pretrained import *
def test_from_pretrained():
    model = from_pretrained("bigscience/T0pp", device_map="auto")
    assert isinstance(model, AutoModelForSeq2SeqLM)
    # Add more test cases here
