# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def complete_sentence_with_model(masked_sentence: str) -> str:
    
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    input_tokens = tokenizer.encode(masked_sentence, return_tensors='pt')
    output_logits = model(input_tokens).logits
    top_predicted_word = tokenizer.decode(output_logits.argmax(-1).flatten())
    
    filled_sentence = masked_sentence.replace('[MASK]', top_predicted_word)
    
    return filled_sentence

# test_function_code --------------------

def test_complete_sentence_with_model():
    print('Testing started.')

    # Test case 1
    print('Testing case [1/2] started.')
    sentence_1 = 'The patient showed signs of fever and a [MASK] heart rate.'
    expected_1 = 'The patient showed signs of fever and a normal heart rate.' # this is an assumed correct output
    result_1 = complete_sentence_with_model(sentence_1)
    assert expected_1 == result_1, f'Test case [1/2] failed: expected {expected_1}, got {result_1}'

    # Test case 2
    print('Testing case [2/2] started.')
    sentence_2 = 'Pulmonary function tests indicate a [MASK] restriction pattern.'
    expected_2 = 'Pulmonary function tests indicate a severe restriction pattern.' # this is an assumed correct output
    result_2 = complete_sentence_with_model(sentence_2)
    assert expected_2 == result_2, f'Test case [2/2] failed: expected {expected_2}, got {result_2}'

    print('Testing finished.')

# call_test_function_line --------------------

test_complete_sentence_with_model()