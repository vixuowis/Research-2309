# requirements_file --------------------

!pip install -U transformers, torch

# function_import --------------------

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

# function_code --------------------

def engage_with_website_visitor(message):
    # Load the Blenderbot model and tokenizer
    model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
    tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')

    # Tokenize the input message from the website visitor
    inputs = tokenizer(message, return_tensors="pt")

    # Use the model to generate a response to the message
    outputs = model.generate(**inputs)

    # Decode the response from the token IDs
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the decoded response
    return response

# test_function_code --------------------

def test_engage_with_website_visitor():
    print("Testing engage_with_website_visitor function.")

    # Test case 1: Standard user question
    question = "How can I cancel my subscription?"
    response = engage_with_website_visitor(question)
    print(f"Response to '{{question}}': {{response}}")
    assert response, "The function did not return a response."

    # Test case 2: Check for non-empty string
    question = "What are your business hours?"
    response = engage_with_website_visitor(question)
    print(f"Response to '{{question}}': {{response}}")
    assert isinstance(response, str) and len(response) > 0, "The function returned an empty or non-string response."

    print("All tests passed for engage_with_website_visitor.")