from f00524_train_model import *
def test_train_model():
	model = Model()
	train_dataset = Dataset()
	eval_dataset = Dataset()
	train_model(model, train_dataset, eval_dataset)


test_train_model()
