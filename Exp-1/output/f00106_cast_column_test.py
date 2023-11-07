from f00106_cast_column import *
def test_cast_column(self):
    audio = Audio(sampling_rate=16_000)
    dataset = self.dataset
    column = "audio"
    casted_dataset = dataset.cast_column(column, audio)
    assert casted_dataset[column].dtype == audio
    assert casted_dataset[column].shape[1] == audio.shape[1]
    assert column not in casted_dataset.column_names
    assert dataset.column_names == casted_dataset.column_names + [column]

