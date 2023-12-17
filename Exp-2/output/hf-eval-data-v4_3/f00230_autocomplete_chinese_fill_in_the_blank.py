# requirements_file --------------------

import subprocess

requirements = ["transformers", "datasets"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForMaskedLM

# function_code --------------------

def autocomplete_chinese_fill_in_the_blank(text, mask_token='[MASK]'):
    """
    Autocompletes the masked token in a Chinese text input using a pre-trained BERT model.

    Args:
        text (str): The input text containing a mask token.
        mask_token (str): The token used for masking a word in the text (default: [MASK]).

    Returns:
        str: The input text with the mask token replaced by the predicted word.

    Raises:
        ValueError: If the `text` does not contain the `mask_token`.
    """
    if mask_token not in text:
        raise ValueError(f'The `mask_token` must be present in the `text`.')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModelForMaskedLM.from_pretrained('bert-base-chinese')
    input_ids = tokenizer.encode(text, return_tensors='pt')
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        output = model(input_ids)
    predictions = output[0]

    predicted_index = torch.argmax(predictions[0, mask_token_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    return text.replace(mask_token, predicted_token)

# test_function_code --------------------

from datasets import load_dataset

def test_autocomplete_chinese_fill_in_the_blank():
    print('Testing started.')
    dataset = load_dataset('chinese_poem')
    sample_data = dataset[0]['text']  # Extract a text sample from the dataset

    # Test case 1: Successful replacement of mask token
    print('Testing case [1/3] started.')
    result = autocomplete_chinese_fill_in_the_blank(f'{sample_data} {mask_token}')
    assert mask_token not in result, f'Test case [1/3] failed: Mask token not replaced.'

    # Test case 2: Confirming the functionality with known masked text
    print('Testing case [2/3] started.')
    result = autocomplete_chinese_fill_in_the_blank('北京是[MASK]国的首都。', mask_token)
    assert '中国' in result, 'Test case [2/3] failed: Incorrect prediction.'

    # Test case 3: Raise an exception when mask token is missing
    print('Testing case [3/3] started.')
    try:
        _ = autocomplete_chinese_fill_in_the_blank('北京是中国的首都。', mask_token)
        assert False, 'Test case [3/3] failed: ValueError exception not raised.'
    except ValueError as e:
        assert str(e) == 'The `mask_token` must be present in the `text`.', f'Test case [3/3] failed: {e}'
    print('Testing finished.')

# call_test_function_line --------------------

test_autocomplete_chinese_fill_in_the_blank()