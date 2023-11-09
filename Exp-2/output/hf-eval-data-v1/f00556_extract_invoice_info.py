from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer
import torch

# Function to extract invoice information
# This function uses a pre-trained model from Hugging Face Transformers to extract specific information from invoices.
# The model is a fine-tuned version of LayoutLMv2 for multimodal document question answering tasks.
def extract_invoice_info(doc_text, question):
    # Load the pre-trained model and tokenizer
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa')

    # Process the invoice document and question with the tokenizer
    inputs = tokenizer(doc_text, question, return_tensors='pt')

    # Perform inference using the model
    outputs = model(**inputs)

    # Post-process the output to obtain the answer
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer