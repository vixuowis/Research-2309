from f00214_pipeline import *
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

# Test Case 1
text = "This is a great product!"
result = classifier(text)
assert result == [{'label': 'POSITIVE', 'score': 0.9994940757751465}] 

# Test Case 2
text = "I am really happy with the service."
result = classifier(text)
assert result == [{'label': 'POSITIVE', 'score': 0.9994940757751465}] 

# Test Case 3
text = "The movie was terrible."
result = classifier(text)
assert result == [{'label': 'NEGATIVE', 'score': 0.9994940757751465}] 

# Test Case 4
text = "The food was delicious."
result = classifier(text)
assert result == [{'label': 'POSITIVE', 'score': 0.9994940757751465}] 

# Test Case 5
text = "I would not recommend this product."
result = classifier(text)
assert result == [{'label': 'NEGATIVE', 'score': 0.9994940757751465}] 
