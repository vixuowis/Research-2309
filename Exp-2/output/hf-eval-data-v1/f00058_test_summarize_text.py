def test_summarize_text():
    # Test the summarize_text function with a sample text
    text = 'Over the past week, the World Health Organization held a conference discussing the impacts of climate change on human health. The conference brought together leading experts from around the world to examine the current problems affecting people's health due to changing environmental conditions. The topics of discussion included increased occurrence of heat-related illnesses, heightened rates of vector-borne diseases, and the growing problem of air pollution. The conference concluded with a call to action for governments and organizations to invest in mitigating and adapting to the negative consequences of climate change for the sake of public health.'
    summary = summarize_text(text)
    # Assert that the summary is not empty
    assert len(summary) > 0
    # Assert that the summary is shorter than the original text
    assert len(summary) < len(text)

test_summarize_text()