from f00838_document_question_answering import *
url = "https://datasets-server.huggingface.co/assets/hf-internal-testing/example-documents/--/hf-internal-testing--example-documents/test/2/image/image.jpg"
question = "What is the total amount?"

result = document_question_answering(url, question)
assert result == {'score': 0.8531, 'answer': '17,000', 'start': 4, 'end': 4}
