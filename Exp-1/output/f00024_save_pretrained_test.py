from f00024_save_pretrained import *
def test_save_pretrained():
    save_directory = "./tf_save_pretrained"
    tokenizer.save_pretrained(save_directory)
    tf_model.save_pretrained(save_directory)


test_save_pretrained()
