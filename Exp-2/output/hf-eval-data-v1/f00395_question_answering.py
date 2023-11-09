from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Function to answer questions using a pre-trained model
# @param question: The question to be answered
# @param context: The context in which the question is asked
# @return: The answer to the question

def question_answering(question: str, context: str) -> str:
    # Load the pre-trained model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained('deepset/deberta-v3-large-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/deberta-v3-large-squad2')

    # Prepare the inputs for the model
    inputs = tokenizer(question, context, return_tensors='pt', max_length=512, truncation=True)

    # Get the model's output
    output = model(**inputs)

    # Get the start and end indices of the answer
    answer_start = output.start_logits.argmax().item()
    answer_end = output.end_logits.argmax().item()

    # Convert the answer from token ids to string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end+1]))

    return answer