from transformers import LayoutLMv3ForQuestionAnswering, LayoutLMv3Tokenizer


def extract_info_from_document(document_path, questions):
    """
    This function extracts information from a scanned document using the LayoutLMv3ForQuestionAnswering model.
    
    Parameters:
    document_path (str): The path to the scanned document.
    questions (list): A list of questions to be answered based on the document.
    
    Returns:
    dict: A dictionary where the keys are the questions and the values are the answers.
    """
    # Initialize the tokenizer and model
    tokenizer = LayoutLMv3Tokenizer.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')
    model = LayoutLMv3ForQuestionAnswering.from_pretrained('hf-tiny-model-private/tiny-random-LayoutLMv3ForQuestionAnswering')

    answers = {}
    # Prepare inputs and pass them to the model
    for question in questions:
        input_data = tokenizer(question, document_path, return_tensors="pt")
        output = model(**input_data)
        answer = tokenizer.convert_ids_to_tokens(output.start_logits.argmax(), output.end_logits.argmax() + 1)
        answers[question] = ' '.join(answer)
    return answers