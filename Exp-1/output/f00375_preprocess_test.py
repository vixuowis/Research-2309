from f00375_preprocess import *
def test_preprocess():
    text = "This is an example text."
    summary = "This is an example summary."
    input_ids, summary_ids = preprocess(text, summary)
    assert isinstance(input_ids, list)
    assert isinstance(summary_ids, list)
    assert all(isinstance(token_id, int) for token_id in input_ids)
    assert all(isinstance(token_id, int) for token_id in summary_ids)


test_preprocess()
