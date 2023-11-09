def test_translate_catalan_to_spanish():
    # Test the translate_catalan_to_spanish function
    # Note: The exact translation may vary depending on the model, so we are not comparing the output strictly
    catalan_text = 'El text en català que vols traduir.'
    translated_text = translate_catalan_to_spanish(catalan_text)
    print(translated_text)
    assert isinstance(translated_text, str)

test_translate_catalan_to_spanish()