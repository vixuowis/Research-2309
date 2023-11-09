from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

def get_best_attractions(question):
    """
    This function uses the DPRQuestionEncoder model to process a user's question and return the most relevant tourist attractions.

    Args:
        question (str): The user's question.

    Returns:
        Tensor: The passage embedding that closely matches the meaning of the user's query.
    """
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

    input_ids = tokenizer(question, return_tensors='pt')['input_ids']
    question_embedding = model(input_ids).pooler_output
    return question_embedding