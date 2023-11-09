def test_generate_chatbot_response():
    instruction = 'Tell me about rose gardening'
    knowledge = 'Roses need well-drained soil and plenty of sun.'
    dialog = ['Hello, how can I assist you with gardening today?', 'I would like to know about rose gardening.']
    output = generate_chatbot_response(instruction, knowledge, dialog)
    assert isinstance(output, str), 'The output should be a string.'
    assert len(output) > 0, 'The output should not be an empty string.'

test_generate_chatbot_response()