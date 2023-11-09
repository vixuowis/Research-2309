from transformers import pipeline, AutoModel, AutoTokenizer


def extract_answer(manual_content: str, question: str) -> str:
    """
    This function uses a pre-trained model to extract answers from a given context.

    Args:
        manual_content (str): The context from which to extract the answer. This should be a string containing the content of the product manual.
        question (str): The question to answer. This should be a string.

    Returns:
        str: The extracted answer.
    """
    qa_pipeline = pipeline(
        'question-answering',
        model=AutoModel.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2'),
        tokenizer=AutoTokenizer.from_pretrained('deepset/bert-large-uncased-whole-word-masking-squad2')
    )

    input_data = {'question': question, 'context': manual_content}
    answer = qa_pipeline(input_data)

    return answer['answer']