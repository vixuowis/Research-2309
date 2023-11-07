from f00849_train_model import *
def test_train_model():
	model = create_model()
	ds = create_dataset()
	result = train_model(model, ds)
	assert result is not None

if __name__ == '__main__':
	test_train_model()
