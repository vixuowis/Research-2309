from f00832_token_classification import *
text = 'Hugging Face is a French company based in New York City.'

result = token_classification(text)

expected_output = [{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}, {'entity': 'I-ORG', 'sco...],
    {'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}]

assert result == expected_output
