from f00015_tokenizer import *
texts = ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."]

expected_output = {
    'input_ids': [[101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 100], [101, 2057, 3246, 2017, 2123, 1005, 1056, 5223, 2009, 1012, 102]],
    'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]
}

assert tokenizer(texts) == expected_output
