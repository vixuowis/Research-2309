from typing import *
from document_qa import document_qa
from image_generator import image_generator

def find_oldest_person(document):
	# Use document_qa to find the oldest person
	answer = document_qa(document, question="What is the oldest person?")
	return answer
