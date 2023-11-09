from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering

def get_answer_from_document(document_text: str, question_text: str) -> str:
    """
    This function takes a document text and a question text as input, and returns the answer to the question based on the document.
    It uses a pre-trained model for document question answering from Hugging Face Transformers.
    
    Parameters:
    document_text (str): The text of the document.
    question_text (str): The text of the question.
    
    Returns:
    str: The answer to the question based on the document.
    """
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    inputs = tokenizer(document_text, question_text, return_tensors='pt')
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax(dim=-1).item()
    answer_end = outputs.end_logits.argmax(dim=-1).item() + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer