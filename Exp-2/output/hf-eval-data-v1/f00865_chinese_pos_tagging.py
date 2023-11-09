from transformers import BertTokenizerFast, AutoModel

def chinese_pos_tagging(chinese_sentence):
    """
    This function performs part-of-speech tagging on a given Chinese sentence.

    Args:
        chinese_sentence (str): The Chinese sentence to be tagged.

    Returns:
        part_of_speech_tags (torch.Tensor): The part-of-speech tags for all tokens in the sentence.
    """
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-pos')

    tokens = tokenizer(chinese_sentence, return_tensors='pt')
    outputs = model(**tokens)
    part_of_speech_tags = outputs.logits.argmax(-1)

    return part_of_speech_tags