from typing import *
from document_qa import document_qa
from image_generator import image_generator

def find_oldest_person(document):
	"""
	Find the oldest person in the document and create an image showcasing the result as a banner.
	
	Args:
		document (str): The document to search for the oldest person.
	
	Returns:
		None
	"""
	answer = document_qa(document, question='What is the oldest person?')
	print(f'The answer is {answer}.')
	image = image_generator('A banner showing ' + answer)
