from transformers import TapasForQuestionAnswering, TapasTokenizer


def table_question_answering(table, query):
    """
    This function uses the TAPAS model from Hugging Face's transformers library to answer questions based on a given table.

    Args:
        table (pd.DataFrame): The table to be queried.
        query (str): The question to be answered based on the table.

    Returns:
        str: The answer to the question.
    """
    model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wtq')
    tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')

    inputs = tokenizer(table=table, queries=query, padding='max_length', return_tensors='pt')
    outputs = model(**inputs)

    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs,
        outputs.logits.detach(),
        outputs.logits_aggregation.detach())

    # let's print out the results:
    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # only a single cell:
            answers.append(table.iat[coordinates[0]])
        else:
            # multiple cells:
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            answers.append(', '.join(cell_values))

    return answers[0]