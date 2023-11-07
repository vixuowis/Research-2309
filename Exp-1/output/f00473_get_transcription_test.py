from f00473_get_transcription import *
def test_get_transcription():
	logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
	processor = Processor()
	transcription = get_transcription(logits, processor)
	assert transcription == ['I WOUL LIKE O SET UP JOINT ACOUNT WTH Y PARTNER']

if __name__ == '__main__':
	test_get_transcription()
