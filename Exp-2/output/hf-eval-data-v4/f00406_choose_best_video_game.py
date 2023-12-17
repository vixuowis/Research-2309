# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def choose_best_video_game(instruction, knowledge, dialog):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = AutoModelForSeq2SeqLM.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog_text = ' EOS '.join(dialog)
    query = f'{instruction} [CONTEXT] {dialog_text} {knowledge}'
    input_ids = tokenizer(query, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

# test_function_code --------------------

def test_choose_best_video_game():
    print('Testing started.')
    # Test case 1
    instruction = 'what is the best way to choose a video game?'
    knowledge = 'Some factors to consider when choosing a video game are personal preferences, genre, graphics, gameplay, storyline, platform, and reviews.'
    dialog = ['What type of video games do you prefer playing?', 'I enjoy action-adventure games and a decent storyline.']
    expected_output = 'A good way to find the best video game for you is to first evaluate your personal preferences and narrow down your options according to the genre, in your case, action-adventure games with a decent storyline. Next, consider the graphics and gameplay. Make sure to read reviews to make your final decision.'
    print('Testing case [1/1] started.')
    result = choose_best_video_game(instruction, knowledge, dialog)
    assert expected_output in result, f'Test case [1/1] failed: Expected part of the output not found.\nExpected: {expected_output}\nResult: {result}'
    print('Test case [1/1] passed.')
    print('Testing finished.')