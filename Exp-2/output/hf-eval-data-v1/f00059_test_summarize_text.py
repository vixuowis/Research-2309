def test_summarize_text():
    '''
    This function tests the summarize_text function.
    It uses a sample text and compares the output of the function to the expected output.
    '''
    sample_text = 'A new study suggests that eating chocolate at least once a week can lead to better cognition. The study, published in the journal Appetite, analyzed data from over 900 adults and found that individuals who consumed chocolate at least once a week performed better on cognitive tests than those who consumed chocolate less frequently. Researchers believe that the beneficial effects of chocolate on cognition may be due to the presence of flavonoids, which have been shown to be antioxidant-rich and to improve brain blood flow.'
    expected_output = 'Eating chocolate at least once a week can lead to better cognition, according to a new study. The research, published in the journal Appetite, analyzed data from over 900 adults and found that those who consumed chocolate at least once a week performed better on cognitive tests. The beneficial effects of chocolate on cognition may be due to the presence of flavonoids, which are antioxidant-rich and improve brain blood flow.'
    assert summarize_text(sample_text) == expected_output

test_summarize_text()