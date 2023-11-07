from f00012_AutoTokenizer.from_pretrained import *
def test_AutoTokenizer_from_pretrained():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    assert isinstance(tokenizer, PreTrainedTokenizer)
    assert isinstance(tokenizer, AutoTokenizer)
