from f00433_remove_columns import *
def test_remove_columns():
    dataset = Dataset.from_dict({
        "audio": ["audio1.wav", "audio2.wav", "audio3.wav"],
        "intent_class": ["intent1", "intent2", "intent3"],
        "speaker_id": [1, 2, 3]
    })
    columns = ["intent_class", "speaker_id"]
    modified_dataset = remove_columns(dataset, columns)
    assert modified_dataset.column_names == ["audio"]
    assert modified_dataset["audio"] == ["audio1.wav", "audio2.wav", "audio3.wav"]

test_remove_columns()
