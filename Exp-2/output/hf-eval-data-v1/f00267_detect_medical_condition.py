from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Function to detect medical condition from an image
# Uses the pretrained 'microsoft/git-large-textvqa' model from Hugging Face Transformers
# The model is fine-tuned on TextVQA and can be used for multimodal tasks like visual question answering
# The function takes an image as input and returns the detected medical condition as output
def detect_medical_condition(image):
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/git-large-textvqa')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/git-large-textvqa')
    encoded_input = tokenizer('What medical condition is present in the image?', image, return_tensors='pt')
    generated_tokens = model.generate(**encoded_input)
    detected_medical_condition = tokenizer.decode(generated_tokens[0])
    return detected_medical_condition