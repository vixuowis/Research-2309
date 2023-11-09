def test_summarize_diary():
    """
    Tests the function 'summarize_diary'.
    """
    diary_entry = 'Today was a great day. I managed to fix the issue with the oxygen generator and had a successful communication session with the ground control. The view of Earth from here is breathtaking.'
    summary = summarize_diary(diary_entry)
    assert isinstance(summary, str), 'The output should be a string.'
    assert len(summary) < len(diary_entry), 'The summary should be shorter than the original text.'

test_summarize_diary()