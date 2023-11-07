from f00447_run_audio_classification import *
def test_run_audio_classification():
	assert run_audio_classification("audio.wav") == [
		{'score': 0.09766869246959686, 'label': 'cash_deposit'},
		{'score': 0.07998877018690109, 'label': 'app_error'},
		{'score': 0.0781070664525032, 'label': 'joint_account'},
		{'score': 0.07667109370231628, 'label': 'pay_bill'},
		{'score': 0.0755252093076706, 'label': 'balance'}
	]

