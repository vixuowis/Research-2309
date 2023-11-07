from f00808_run_xla_forward_pass import *
def test_run_xla_forward_pass():
	assert run_xla_forward_pass(model, random_inputs) == None


def main():
	test_run_xla_forward_pass()


if __name__ == '__main__':
	main()

