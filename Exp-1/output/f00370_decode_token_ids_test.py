from f00370_decode_token_ids import *
tokenizer = AutoTokenizer.from_pretrained('model_name')
token_ids = [101, 1234, 5678, 102]

def test_decode_token_ids():
    assert decode_token_ids(tokenizer, token_ids) == 'Les lugumes partagent les ressources avec des bactéries fixatrices d'azote.'


def test_all():
    test_decode_token_ids()


test_all()
