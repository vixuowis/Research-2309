from f00213_run_inference import *
def test_run_inference():
	text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
	expected_sentiment = "positive"
	result = run_inference(text)
	assert result == expected_sentiment, f'Expected sentiment: {expected_sentiment}, Got: {result}'

	text = "This movie was terrible. Acting was bad and the plot was confusing. Would not recommend."
	expected_sentiment = "negative"
	result = run_inference(text)
	assert result == expected_sentiment, f'Expected sentiment: {expected_sentiment}, Got: {result}'

	text = "The food at this restaurant was amazing. Great flavors and presentation. Highly recommend."
	expected_sentiment = "positive"
	result = run_inference(text)
	assert result == expected_sentiment, f'Expected sentiment: {expected_sentiment}, Got: {result}'

	text = "The customer service was terrible. Rude staff and long wait times. Would not visit again."
	expected_sentiment = "negative"
	result = run_inference(text)
	assert result == expected_sentiment, f'Expected sentiment: {expected_sentiment}, Got: {result}'

	text = "The product was average. Not great, but not terrible either."
	expected_sentiment = "neutral"
	result = run_inference(text)
	assert result == expected_sentiment, f'Expected sentiment: {expected_sentiment}, Got: {result}'
