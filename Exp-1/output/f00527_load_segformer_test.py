from f00527_load_segformer import *
def test_load_segformer():
    checkpoint = 'segformer_checkpoint'
    id2label = {0: 'background', 1: 'object'}
    label2id = {'background': 0, 'object': 1}
    optimizer = tf.keras.optimizers.Adam()
    model = load_segformer(checkpoint, id2label, label2id, optimizer)
    assert isinstance(model, TFAutoModelForSemanticSegmentation)


test_load_segformer()
