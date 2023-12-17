# requirements_file --------------------

!pip install -U transformers torch datasets tokenizers

# function_import --------------------

from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# function_code --------------------

def extract_answers_from_docs(question, document):
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')

    inputs = tokenizer(question, document, return_tensors="pt")
    outputs = model(**inputs)
    start_position = outputs.start_logits.argmax().item()
    end_position = outputs.end_logits.argmax().item()
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_position:end_position+1]))
    return answer

# test_function_code --------------------

def test_extract_answers_from_docs():
    print("Testing started.")
    doc = "The capital of France is Paris. It is located in Europe and uses the Euro as its currency."
    question = "What is the capital of France?"
    expected_answer = "Paris"

    print("Testing case [1/1] started.")
    assert extract_answers_from_docs(question, doc) == expected_answer, f"Test case [1/1] failed: Expected {expected_answer} but got {extract_answers_from_docs(question, doc)}"
    print("Testing finished.")

test_extract_answers_from_docs()