from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Function to get answer for a given question and context
# Uses the ELECTRA_large_discriminator language model fine-tuned on SQuAD2.0 for question answering tasks.
def get_answer(question: str, context: str) -> str:
    # Load the pre-trained model
    model = AutoModelForQuestionAnswering.from_pretrained('ahotrod/electra_large_discriminator_squad2_512')
    # Create a tokenizer instance
    tokenizer = AutoTokenizer.from_pretrained('ahotrod/electra_large_discriminator_squad2_512')
    # Convert the question and context into input format suitable for the model
    inputs = tokenizer(question, context, return_tensors='pt')
    # Pass the tokenized inputs to the model for inference
    outputs = model(**inputs)
    # Decode the produced logits into a human-readable answer
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    return answer