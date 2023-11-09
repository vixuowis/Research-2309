from transformers import BertTokenizer, AlbertForMaskedLM, FillMaskPipeline

def fill_mask_chinese(sentence):
    """
    This function fills in the [MASK] token in a given Chinese sentence using the 'uer/albert-base-chinese-cluecorpussmall' pre-trained model.
    
    Parameters:
    sentence (str): The Chinese sentence with a [MASK] token.
    
    Returns:
    str: The sentence with the [MASK] token replaced by the most probable word according to the model.
    """
    tokenizer = BertTokenizer.from_pretrained('uer/albert-base-chinese-cluecorpussmall')
    model = AlbertForMaskedLM.from_pretrained('uer/albert-base-chinese-cluecorpussmall')
    unmasker = FillMaskPipeline(model, tokenizer)
    filled_sentence = unmasker(sentence)[0]['sequence']
    return filled_sentence