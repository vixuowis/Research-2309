from transformers import AutoModelForDocumentQuestionAnswering, AutoTokenizer

# Function to answer questions from scanned documents
# Uses the Hugging Face Transformers library and a pre-trained model
# The model is 'tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa'
def document_question_answer(question, scanned_document_text):
    # Load the pre-trained model and tokenizer
    model = AutoModelForDocumentQuestionAnswering.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')
    tokenizer = AutoTokenizer.from_pretrained('tiennvcs/layoutlmv2-base-uncased-finetuned-infovqa')

    # Tokenize the question and scanned document text
    inputs = tokenizer(question, scanned_document_text, return_tensors='pt')

    # Feed the inputs to the model
    output = model(**inputs)

    # Return the answer
    return output