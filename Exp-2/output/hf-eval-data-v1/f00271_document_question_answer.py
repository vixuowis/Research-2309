from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer
import torch

# Function to answer questions based on the content of a given document
# Uses the Hugging Face Transformers library and a pre-trained model

def document_question_answer(document_content, question):
    # Load the pre-trained model
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    # Tokenize the input document and the question
    inputs = tokenizer(document_content, question, return_tensors='pt', padding='max_length', max_length=512, truncation='only_first')
    # Run the model to receive an answer
    outputs = model(**inputs)
    # Get the answer from the model's output
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer
    answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer