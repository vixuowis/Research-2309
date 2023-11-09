# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def fill_mask_with_bio_clinical_bert(masked_sentence):
    """
    This function fills the mask token in a given sentence using the Bio_ClinicalBERT model.

    Args:
        masked_sentence (str): The sentence with a missing word, represented by the [MASK] token.

    Returns:
        str: The sentence with the mask token replaced by the most probable word predicted by the Bio_ClinicalBERT model.
    """
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    input_tokens = tokenizer.encode(masked_sentence, return_tensors="pt")
    output_logits = model(input_tokens).logits
    top_predicted_word = tokenizer.decode(output_logits.argmax(-1).item())
    filled_sentence = masked_sentence.replace("[MASK]", top_predicted_word)
    return filled_sentence

# test_function_code --------------------

def test_fill_mask_with_bio_clinical_bert():
    """
    This function tests the fill_mask_with_bio_clinical_bert function by comparing the output with the expected result.
    """
    test_sentence = "The patient showed signs of fever and a [MASK] heart rate."
    expected_result = "The patient showed signs of fever and a high heart rate."
    assert fill_mask_with_bio_clinical_bert(test_sentence) == expected_result

# call_test_function_code --------------------

test_fill_mask_with_bio_clinical_bert()