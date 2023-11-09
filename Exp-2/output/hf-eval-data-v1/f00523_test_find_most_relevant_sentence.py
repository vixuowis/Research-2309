def test_find_most_relevant_sentence():
    question = 'What is the main purpose of photosynthesis?'
    sentences = ['Photosynthesis is the process used by plants to convert light energy into chemical energy to fuel their growth.', 'The Eiffel Tower is a famous landmark in Paris.', 'Photosynthesis also produces oxygen as a byproduct, which is necessary for life on Earth.']
    assert find_most_relevant_sentence(question, sentences) == 'Photosynthesis is the process used by plants to convert light energy into chemical energy to fuel their growth.'
    
    question = 'Where is the Eiffel Tower located?'
    sentences = ['The Eiffel Tower is in Paris.', 'The Statue of Liberty is in New York.', 'The Leaning Tower of Pisa is in Italy.']
    assert find_most_relevant_sentence(question, sentences) == 'The Eiffel Tower is in Paris.'
    
    question = 'What does the Leaning Tower of Pisa famous for?'
    sentences = ['The Leaning Tower of Pisa is famous for its tilt.', 'The Eiffel Tower is famous for its height.', 'The Statue of Liberty is famous for its torch.']
    assert find_most_relevant_sentence(question, sentences) == 'The Leaning Tower of Pisa is famous for its tilt.'

test_find_most_relevant_sentence()