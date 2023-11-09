from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering


def document_question_answer(question: str, document: str) -> str:
    """
    This function takes a question and a document as input and returns the answer to the question based on the document.
    It uses a pre-trained model from Hugging Face Transformers.
    
    Args:
        question (str): The question to be answered.
        document (str): The document to find the answer from.
    
    Returns:
        str: The answer to the question.
    """
    # Load the pre-trained model
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    
    # Load the corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    
    # Tokenize the input document and question
    input_dict = tokenizer(question, document, return_tensors='pt')
    
    # Pass the encoded input to the model to get the answer
    output = model(**input_dict)
    
    # Convert the answer from ids to tokens
    answer = tokenizer.convert_ids_to_tokens(output['answer_ids'][0])
    
    return answer