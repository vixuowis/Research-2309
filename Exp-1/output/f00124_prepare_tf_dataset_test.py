from f00124_prepare_tf_dataset import *
def test_prepare_tf_dataset():
    dataset = load_dataset('imdb', split='train[:10]')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tf_dataset = prepare_tf_dataset(dataset, batch_size=16, shuffle=True, tokenizer=tokenizer)
    assert tf_dataset is not None
    assert isinstance(tf_dataset, tf.data.Dataset)
    assert len(list(tf_dataset)) == 1


test_prepare_tf_dataset()
