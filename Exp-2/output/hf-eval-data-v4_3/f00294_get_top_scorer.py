# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import json
from transformers import TapasForQuestionAnswering, TapasTokenizer

# function_code --------------------

def get_top_scorer(question: str, data: str) -> str:
        """
        Identify the player with the most goals scored in a given match.

        Args:
            question (str): The question in natural language form.
            data (str): The game statistics in CSV format as a string.

        Returns:
            str: The name of the player who scored the most goals.

        Raises:
            ValueError: If the question or data is empty or not in the proper format.
        """
        if not question or not data:
            raise ValueError("Question and data cannot be empty")

        # Load the pre-trained TAPAS model specifically fine-tuned for question answering
        model = TapasForQuestionAnswering.from_pretrained('google/tapas-large-finetuned-sqa')
        tokenizer = TapasTokenizer.from_pretrained('google/tapas-large-finetuned-sqa')

        # Tokenize the input question and table data
        inputs = tokenizer(question, data, return_tensors="pt")
        outputs = model(**inputs)

        # Extract the top answer from the outputs
        answer_index = outputs.logits.argmax(axis=1).item()
        answer_label = tokenizer.convert_ids_to_tokens([answer_index])[0]

        # Parse the answer and extract the player's name
        answer_data = json.loads(data)
        top_scorer = next((player for player, goals in answer_data.items() if player in answer_label), 'N/A')

        return top_scorer

# test_function_code --------------------

def test_get_top_scorer():
    print("Testing started.")
    # Prepare the test data and expected results
    question = "What player scored the most goals?"
    data = "Player,Goals\nA,2\nB,3\nC,1"
    expected_name = "B"

    # Test case 1: Valid input
    print("Testing case [1/1] started.")
    result = get_top_scorer(question, data)
    assert result == expected_name, f"Test case [1/1] failed: Expected {expected_name}, got {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_get_top_scorer()