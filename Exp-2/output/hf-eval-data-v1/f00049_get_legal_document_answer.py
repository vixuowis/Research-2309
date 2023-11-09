from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# This function is designed to extract answers from legal contracts and documents
# It uses the Rakib/roberta-base-on-cuad model from Hugging Face Transformers, which is trained on the CUAD dataset
# The function takes a question and a context (the legal document) as input and returns the answer

def get_legal_document_answer(question: str, context: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained('Rakib/roberta-base-on-cuad')
    model = AutoModelForQuestionAnswering.from_pretrained('Rakib/roberta-base-on-cuad')

    inputs = tokenizer(question, context, return_tensors='pt')
    outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))