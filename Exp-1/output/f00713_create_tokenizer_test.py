from f00713_create_tokenizer import *
def test_create_tokenizer():
    tokenizer = create_tokenizer("distilbert-base-uncased")
    assert isinstance(tokenizer, DistilBertTokenizer)

    # Add more test cases if needed

if __name__ == "__main__":
    test_create_tokenizer()
