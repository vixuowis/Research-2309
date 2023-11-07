from f00455_remove_columns import *
def test_remove_columns():
    dataset = Dataset({
        "audio": ["audio1.wav", "audio2.wav", "audio3.wav"],
        "transcription": ["trans1", "trans2", "trans3"]
    })
    columns = ["audio"]
    expected_dataset = Dataset({
        "transcription": ["trans1", "trans2", "trans3"]
    })

    processed_dataset = remove_columns(dataset, columns)

    assert processed_dataset == expected_dataset


test_remove_columns()
