from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering


def extract_answer_from_document(question: str, context: str) -> str:
    """
    This function uses a pretrained LayoutLMv2 model to analyze the text in a document and extract answers to questions based on the content.
    
    Parameters:
    question (str): The question to be answered based on the document.
    context (str): The text of the document.
    
    Returns:
    str: The answer to the question based on the document.
    """
    model_checkpoint = 'L-oenai/LayoutLMX_pt_question_answer_ocrazure_correct_V15_30_03_2023'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)
    inputs = tokenizer.prepare_seq2seq_batch([question], context, return_tensors='pt')
    outputs = model(**inputs)
    ans_start, ans_end = outputs.start_logits.argmax(), outputs.end_logits.argmax()
    answer = tokenizer.decode(inputs['input_ids'][0][ans_start : ans_end + 1])
    return answer