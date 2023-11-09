from transformers import TapasTokenizer, TapasForQuestionAnswering

def table_question_answering(table, question):
    """
    This function receives a table and a question in natural language, and returns an answer to the question based on the inputted table.
    It uses the pretrained Tapas model 'lysandre/tapas-temporary-repo' from Hugging Face Transformers.
    
    Parameters:
    table (str): The input table.
    question (str): The input question in natural language.
    
    Returns:
    str: The answer to the question based on the inputted table.
    """
    # Instantiate a TapasTokenizer and a TapasForQuestionAnswering model
    tokenizer = TapasTokenizer.from_pretrained('lysandre/tapas-temporary-repo')
    model = TapasForQuestionAnswering.from_pretrained('lysandre/tapas-temporary-repo')
    
    # Call tokenizer with the table and the question to get input tensors
    inputs = tokenizer(table=table, queries=question, return_tensors='pt')
    
    # Pass the input tensors through the pretrained Tapas model to get the logits
    outputs = model(**inputs)
    
    # Convert the logits into readable predictions
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach(), outputs.logits_aggregation.detach())
    
    # Extract the answer from the inputted table using the predicted answer coordinates
    answer = ''
    for coordinates in predicted_answer_coordinates:
        for coordinate in coordinates:
            answer += table[coordinate[0]][coordinate[1]] + ' '
    
    return answer.strip()