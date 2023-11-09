from transformers import AutoTokenizer, AutoModelForCausalLM

# Function to generate code based on a natural language description
# Uses the Hugging Face Transformers library and the pretrained model 'Salesforce/codegen-2B-multi'
def generate_code(description):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-2B-multi')
    model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-2B-multi')

    # Convert the description into a format that can be processed by the model
    input_ids = tokenizer(description, return_tensors='pt').input_ids

    # Generate a code snippet based on the description
    generated_ids = model.generate(input_ids, max_length=128)

    # Decode the generated snippet into human-readable text
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_code