def test_analyze_customer_feedback():
    customer_feedback = 'Me encanta este producto!'
    sentiment = analyze_customer_feedback(customer_feedback)
    assert sentiment[0]['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL'], 'Invalid sentiment'

test_analyze_customer_feedback()