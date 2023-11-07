from f00258_preprocess_function import *
examples = {
    "question": ["What is the capital of France?"],
    "context": "The capital of France is Paris.",
    "answers": [{"answer_start": [19], "text": ["Paris"]}],
}

preprocessed_inputs = preprocess_function(examples)
print(preprocessed_inputs)
