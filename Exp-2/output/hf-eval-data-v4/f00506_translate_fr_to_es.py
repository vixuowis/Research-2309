# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def translate_fr_to_es(french_text):
    """
    Translates French text to Spanish using a pre-trained model from Hugging Face.
    
    Args:
    french_text (str): The text in French to be translated to Spanish.
    
    Returns:
    str: The translated text in Spanish.
    """
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    model = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-fr-es')
    inputs = tokenizer(french_text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# test_function_code --------------------

def test_translate_fr_to_es():
    print("Testing translation function started.")
    
    # 测试用例 1：一个简单的问候语
    print("Testing case [1/3] started.")
    french_text = "Bonjour, comment ça va?"
    expected_translation = "Hola, ¿cómo estás?"  # 预期的翻译可能会有出入，这里只是一个近似值
    translated_text = translate_fr_to_es(french_text)
    assert translated_text.startswith(expected_translation), f"Test case [1/3] failed: Expected start of translation to be '{expected_translation}', but got '{translated_text}'."

    # 测试用例 2：一句较为复杂的语句
    print("Testing case [2/3] started.")
    french_text = "Je voudrais réserver une table pour deux personnes, s'il vous plaît."
    translated_text = translate_fr_to_es(french_text)
    # 在这个测试用例里，我们不能直接检查翻译的准确性，因为可能存在多个正确的翻译
    assert isinstance(translated_text, str) and len(translated_text) > 0, f"Test case [2/3] failed: The translation should be a non-empty string."

    # 测试用例 3：空字符串
    print("Testing case [3/3] started.")
    french_text = ""
    translated_text = translate_fr_to_es(french_text)
    assert translated_text == "", f"Test case [3/3] failed: Expected translation of an empty string to be empty, but got '{translated_text}'."
    
    print("Testing finished.")

# 运行测试函数
test_translate_fr_to_es()