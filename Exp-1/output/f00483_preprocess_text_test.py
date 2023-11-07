from f00483_preprocess_text import *
def test_preprocess_text():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    text = 'Hello, how are you?'
    expected_output = ['hello', ',', 'how', 'are', 'you', '?']
    assert preprocess_text(text, tokenizer) == expected_output
    
    text = 'I am doing great!'
    expected_output = ['i', 'am', 'doing', 'great', '!']
    assert preprocess_text(text, tokenizer) == expected_output
    
    text = 'This is a test.'
    expected_output = ['this', 'is', 'a', 'test', '.']
    assert preprocess_text(text, tokenizer) == expected_output
    
    text = 'Another example text'
    expected_output = ['another', 'example', 'text']
    assert preprocess_text(text, tokenizer) == expected_output
    
    text = 'One more example'
    expected_output = ['one', 'more', 'example']
    assert preprocess_text(text, tokenizer) == expected_output
    
    print('All test cases pass!')

test_preprocess_text()
