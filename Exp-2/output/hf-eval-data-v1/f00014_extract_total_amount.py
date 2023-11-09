from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# Function to extract total amount from invoice document
# Uses Hugging Face Transformers and a pre-trained model 'impira/layoutlm-invoices'
def extract_total_amount(question: str, context: str) -> str:
    # Load the pre-trained model
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('impira/layoutlm-invoices')
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('impira/layoutlm-invoices')
    # Encode the input question and context
    inputs = tokenizer(question, context, return_tensors='pt')
    # Pass the encoded input to the model
    outputs = model(**inputs)
    # Obtain the answer by taking the highest scoring tokens
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item()
    # Decode the answer tokens back to a textual answer
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start: answer_end + 1].tolist()))
    return answer