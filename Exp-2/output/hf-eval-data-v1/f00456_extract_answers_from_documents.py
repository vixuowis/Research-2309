from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer
import torch

# Function to extract answers from documents given a set of questions
# Uses the Hugging Face Transformers library and a fine-tuned model for document question answering
def extract_answers_from_documents(questions: list, document: str) -> dict:
    # Load the fine-tuned model
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')
    # Instantiate a tokenizer
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')

    answers = {}
    for question in questions:
        # Tokenize the question and document
        inputs = tokenizer(question, document, return_tensors='pt')
        # Get the model's output
        outputs = model(**inputs)
        # Get the start and end positions of the answer
        start_position = torch.argmax(outputs.start_logits).item()
        end_position = torch.argmax(outputs.end_logits).item()
        # Retrieve the answer
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_position:end_position+1]))
        answers[question] = answer
    return answers