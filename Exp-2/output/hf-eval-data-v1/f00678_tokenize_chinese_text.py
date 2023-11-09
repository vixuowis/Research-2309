from transformers import BertTokenizerFast, AutoModel


def tokenize_chinese_text(text):
    """
    This function tokenizes Chinese text using the 'ckiplab/bert-base-chinese-ws' pretrained model.
    
    Parameters:
    text (str): The Chinese text to be tokenized.
    
    Returns:
    tokens (list): The tokenized text.
    """
    # Instantiate the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    # Load the pretrained model
    model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ws')
    # Tokenize the text
    tokens = tokenizer(text)
    return tokens