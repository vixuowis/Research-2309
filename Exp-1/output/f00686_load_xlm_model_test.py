from f00686_load_xlm_model import *
def test_load_xlm_model():
    tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
    model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")
    assert isinstance(tokenizer, XLMTokenizer)
    assert isinstance(model, XLMWithLMHeadModel)

if __name__ == "__main__":
    test_load_xlm_model()
