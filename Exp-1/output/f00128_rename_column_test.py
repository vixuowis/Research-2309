from f00128_rename_column import *
def test_rename_column():
    dataset = Dataset.from_dict({"label": [0, 1, 0]})
    renamed_dataset = dataset.rename_column("label", "labels")
    assert renamed_dataset.column_names == ["labels"]
    assert renamed_dataset["labels"] == [0, 1, 0]

test_rename_column()
