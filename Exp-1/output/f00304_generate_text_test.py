from f00304_generate_text import *
def test_generate_text():
	model_name = "my_awesome_eli5_clm-model"
	prompt = "Somatic hypermutation allows the immune system to be able to effectively reverse the damage caused by an infection."
	expected_output = "Somatic hypermutation allows the immune system to be able to effectively reverse the damage caused by an infection.\n\n\nThe damage caused by an infection is caused by the immune system's ability to perform its own self-correcting tasks."
	assert generate_text(model_name, prompt) == expected_output

	# Add more test cases here
	
	print("All test cases pass")
