from transformers import AutoModelForCausalLM


def convert_text_to_speech(text):
    """
    Convert a given text into spoken Japanese using a pre-trained model from ESPnet.

    Args:
        text (str): The text to be converted into speech.

    Returns:
        torch.Tensor: The output tensor from the model, representing the spoken Japanese.
    """
    model = AutoModelForCausalLM.from_pretrained('espnet/kan-bayashi_jvs_tts_finetune_jvs001_jsut_vits_raw_phn_jaconv_pyopenjta-truncated-178804')
    return model(text)