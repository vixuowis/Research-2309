from transformers import pipeline


def extract_medical_info(document_text: str, question: str) -> str:
    '''
    This function uses a fine-tuned version of cambridgeltl/SapBERT-from-PubMedBERT-fulltext on the squad_v2 dataset
    to extract answers to questions from a large medical document.

    Args:
    document_text (str): The large medical document text.
    question (str): The question that needs to be answered.

    Returns:
    str: The answer to the question.
    '''
    # Create a question answering pipeline with the specified model
    qa_pipeline = pipeline('question-answering', model='bigwiz83/sapbert-from-pubmedbert-squad2')

    # Use the pipeline to extract the answer to the question from the document
    answer = qa_pipeline({'context': document_text, 'question': question})

    return answer