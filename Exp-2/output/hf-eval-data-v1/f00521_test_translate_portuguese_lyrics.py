def test_translate_portuguese_lyrics():
    # Define a test case of Portuguese lyrics
    lyrics = 'Tinha uma pedra no meio do caminho'
    # Translate the lyrics
    translated_lyrics = translate_portuguese_lyrics(lyrics)
    # Assert that the translation is not None
    assert translated_lyrics is not None
    # Assert that the translation is a string
    assert isinstance(translated_lyrics, str)
    # Print the translated lyrics for manual inspection
    print('Translated lyrics:', translated_lyrics)

# Run the test function
test_translate_portuguese_lyrics()