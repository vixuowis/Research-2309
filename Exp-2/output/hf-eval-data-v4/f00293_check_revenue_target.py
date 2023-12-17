# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import TapasForQuestionAnswering, TapasTokenizer

# function_code --------------------

def check_revenue_target(table_data, query, target_revenue):
    # Load the TAPAS model and tokenizer
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-small-finetuned-wtq')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-small-finetuned-wtq')

    # Tokenize the input data
    inputs = tokenizer(table=table_data, queries=query, return_tensors='pt')
    # Get predictions from the TAPAS model
    outputs = model(**inputs)

    # Convert logits to predictions
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs,
        outputs.logits.detach(),
        outputs.logits_aggregation.detach()
    )

    # Check if the aggregated sum of revenues matches or exceeds the target
    predicted_sum = sum([table_data['Revenue'][i[0]] for i in predicted_answer_coordinates])
    return predicted_sum >= target_revenue


# test_function_code --------------------

def test_check_revenue_target():
    print("Testing started.")
    table_data = {"Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                  "Revenue": [2000, 2500, 3000, 3500, 4000, 4500, 5000]}
    target_revenue = 24000
    query = "Did the total revenue meet the target revenue of 24000?"

    # Test if the target revenue is met
    assert check_revenue_target(table_data, query, target_revenue), "Test case failed: The total revenue should meet the target."
    print("Testing finished.")

# Run the test function
test_check_revenue_target()
