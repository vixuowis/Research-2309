def test_translate_and_summarize():
    """
    This function tests the translate_and_summarize function by comparing the output with the expected result.
    """
    input_text = 'translate English to German: How old are you?'
    expected_output = 'Wie alt bist du?'
    assert translate_and_summarize(input_text).strip() == expected_output

    input_text = 'summarize: Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn patterns from data without being explicitly programmed. The field has seen tremendous growth in recent years, driven by advances in computational power, the abundance of data, and improvements in algorithms. There are many types of machine learning algorithms, including supervised learning, unsupervised learning, reinforcement learning, and deep learning. Applications of machine learning are diverse, ranging from image and speech recognition to financial trading and recommendation systems.'
    expected_output = 'Machine learning, a subset of artificial intelligence, develops algorithms to learn patterns without explicit programming. Driven by computational advancement, abundant data, and algorithmic improvements, it includes supervised, unsupervised, reinforcement, and deep learning algorithms. Applications span from image and speech recognition to financial trading and recommendation systems.'
    assert translate_and_summarize(input_text).strip() == expected_output