# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import TapasTokenizer, TapasForQuestionAnswering

# function_code --------------------

def get_total_sales_of_product(model_name: str, question: str, table: list) -> float:
    # Load the pre-trained TAPAS tokenizer
    tokenizer = TapasTokenizer.from_pretrained(model_name)

    # Load TAPAS model for table-based question answering
    model = TapasForQuestionAnswering.from_pretrained(model_name)

    # Create inputs tensors for TAPAS using the table and the question
    inputs = tokenizer(table=table, queries=[question], return_tensors='pt')

    # Pass the inputs to the model
    outputs = model(**inputs)

    # Get the predicted answer coordinates and aggregation indices
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs, 
        outputs.logits.detach(), 
        outputs.logits_aggregation.detach()
    )

    # Calculate the sum of sales for the desired product using the predicted aggregation indices
    total_sales = 0
    for idx, aggregation_index in enumerate(predicted_aggregation_indices):
        if aggregation_index == 0:  # 0 corresponds to the SUM operation in TAPAS
            row, col = predicted_answer_coordinates[idx]
            total_sales += float(table[row][col])

    return total_sales

# test_function_code --------------------

def test_get_total_sales_of_product():
    print("Testing get_total_sales_of_product function.")

    # Sample table data
    sample_table = [
        ['Product', 'Week 1', 'Week 2', 'Week 3'],
        ['Product A', '100', '200', '300'],
        ['Product B', '150', '250', '350']
    ]

    # Sample question
    question = "What is the total sales of Product A?"

    # Expected answer
    expected_sales = 600

    # Test case
    total_sales = get_total_sales_of_product('lysandre/tapas-temporary-repo', question, sample_table)
    assert total_sales == expected_sales, f"Test failed: Expected {expected_sales}, got {total_sales}"
    print("Test passed.")

test_get_total_sales_of_product()