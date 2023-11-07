from f00868_create_upernet_model import *
def test_create_upernet_model():
    model = create_upernet_model()
    assert isinstance(model, UperNetForSemanticSegmentation)

    # Add more test cases here

if __name__ == '__main__':
    test_create_upernet_model()
