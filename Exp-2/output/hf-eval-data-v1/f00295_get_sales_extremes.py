from transformers import TapasTokenizer, TapasForQuestionAnswering

def get_sales_extremes(sales_data_table):
    """
    This function uses the TapasForQuestionAnswering model from Hugging Face Transformers to answer the question
    'What are the highest and lowest sales numbers?' based on the provided sales data table.
    
    Args:
        sales_data_table (list): A list of dictionaries representing the sales data table.
    
    Returns:
        tuple: A tuple containing the highest and lowest sales numbers.
    """
    # Load the pre-trained model and tokenizer
    tokenizer = TapasTokenizer.from_pretrained('lysandre/tapas-temporary-repo')
    model = TapasForQuestionAnswering.from_pretrained('lysandre/tapas-temporary-repo')
    
    # Tokenize the input table and query
    inputs = tokenizer(table=sales_data_table, queries="What are the highest and lowest sales numbers?", return_tensors='pt')
    
    # Provide the tokenized inputs to the model
    outputs = model(**inputs)
    
    # Convert the logits into predicted answers and aggregation indices
    predicted_answer_coordinates, _ = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
    
    # Extract the highest and lowest sales numbers from the predicted results
    highest_sales, lowest_sales = extract_aggregated_sales(predicted_answer_coordinates)
    
    return highest_sales, lowest_sales