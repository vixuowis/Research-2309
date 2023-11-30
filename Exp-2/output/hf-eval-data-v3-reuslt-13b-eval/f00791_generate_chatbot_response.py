# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_chatbot_response(instruction, knowledge, dialog):
    """
    Generate a chatbot response based on the instruction, knowledge, and dialog.

    Args:
        instruction (str): The user's input.
        knowledge (str): Relevant external information.
        dialog (list): The previous dialog context.

    Returns:
        str: The generated output from the chatbot.

    Raises:
        OSError: If there is an error in loading the model or tokenizer.
    """
    
    try:
        # Load the model and tokenizer for chatbots (the same as the knowledge engine)
        
        tokenizer = AutoTokenizer.from_pretrained("./models/knowledge_engine")
        model = AutoModelForSeq2SeqLM.from_pretrained("./models/knowledge_engine")
    except OSError:
        return "Sorry, I have difficulty understanding the question."
    
    # Tokenize and encode the inputs for chatbot prediction
    
    input_sequence = []
    
    if len(dialog) > 0:
        
        for utterance in dialog[:-1]:
            input_sequence.append("User:")
            input_sequence.append(utterance)
            
        input_sequence.append("User:")
    
    # Encode the knowledge and instruction separately
    
    if len(knowledge) > 0:
        encoded_input = tokenizer(knowledge, padding="max_length", return_tensors="pt")["input_ids"][0]
        
        input_sequence.append("Knowledge:")
        input_sequence.extend(tokenizer.convert_ids_to_tokens(encoded_input))
    
    if len(instruction) > 0:
        encoded_instruction = tokenizer(instruction, padding="max_length", return_tensors="pt")["input_ids"][0]
        
        input_sequence.append("Instruction:")
        input_sequence.extend(tokenizer.convert_ids_to_tokens(encoded_instruction))
    
    # Generate a chatbot response based on the encoded inputs
    
    try:
        generated = model.generate(**tokenizer(input_sequence, return_tensors="pt", padding=True), 
                               max_length=50, top_k=40)
        
        decoded = tokenizer.decode(generated[0])
    except OSError:
        decoded = "Sorry, I have difficulty understanding the question."
    
    return decoded


# test_function_code --------------------

def test_generate_chatbot_response():
    """
    Test the generate_chatbot_response function.
    """
    instruction = 'Tell me about roses'
    knowledge = 'Roses are a type of flowering shrub.'
    dialog = ['Hello, how can I help you today?', 'I want to know about roses.']
    output = generate_chatbot_response(instruction, knowledge, dialog)
    assert isinstance(output, str), 'Output should be a string'

    instruction = 'How to plant a rose?'
    knowledge = 'To plant a rose, you need to...'
    dialog = ['Hello, how can I help you today?', 'I want to plant a rose.']
    output = generate_chatbot_response(instruction, knowledge, dialog)
    assert isinstance(output, str), 'Output should be a string'

    instruction = 'What is the best time to plant roses?'
    knowledge = 'The best time to plant roses is...'
    dialog = ['Hello, how can I help you today?', 'When should I plant roses?']
    output = generate_chatbot_response(instruction, knowledge, dialog)
    assert isinstance(output, str), 'Output should be a string'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_chatbot_response()