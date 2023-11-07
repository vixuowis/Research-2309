from f00325_evaluate_model import *
def test_evaluate_model():
	trainer = 'dummy_trainer'
	result = evaluate_model(trainer)
	assert result == 8.76, 'Test case 1 failed'
	trainer = 'another_dummy_trainer'
	result = evaluate_model(trainer)
	assert result == 8.76, 'Test case 2 failed'
	trainer = 'yet_another_dummy_trainer'
	result = evaluate_model(trainer)
	assert result == 8.76, 'Test case 3 failed'
	trainer = 'one_more_dummy_trainer'
	result = evaluate_model(trainer)
	assert result == 8.76, 'Test case 4 failed'
	trainer = 'final_dummy_trainer'
	result = evaluate_model(trainer)
	assert result == 8.76, 'Test case 5 failed'


test_evaluate_model()
