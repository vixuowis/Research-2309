from f00367_decode_token_ids import *
tokenizer = T5Tokenizer.from_pretrained('t5-base')
token_ids = [1, 2, 3, 4, 5]

def test_decode_token_ids():
    assert decode_token_ids(tokenizer, token_ids) == 'Les lignées partagent des ressources avec des bactéries enfixant l'azote.'

test_decode_token_ids()
