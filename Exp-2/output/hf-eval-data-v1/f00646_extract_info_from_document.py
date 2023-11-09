from transformers import pipeline


def extract_info_from_document(ocr_extracted_text: str, question: str) -> str:
    """
    This function uses a fine-tuned model from Hugging Face Transformers to extract relevant information from OCR scanned text.
    
    Parameters:
    ocr_extracted_text (str): The text extracted from OCR scanning.
    question (str): The question to be answered based on the OCR text.
    
    Returns:
    str: The answer to the question based on the OCR text.
    """
    # Create a question-answering model
    qa_pipeline = pipeline('question-answering', model='tiennvcs/layoutlmv2-large-uncased-finetuned-vi-infovqa')
    
    # Use the model to answer the question based on the OCR text
    answer = qa_pipeline({"context": ocr_extracted_text, "question": question})
    
    return answer['answer']