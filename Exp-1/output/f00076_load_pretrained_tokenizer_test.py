from f00076_load_pretrained_tokenizer import *
def test_load_pretrained_tokenizer():
    tokenizer = load_pretrained_tokenizer('bert-base-cased')
    assert isinstance(tokenizer, AutoTokenizer)

    print('All test cases pass')

test_load_pretrained_tokenizer()
