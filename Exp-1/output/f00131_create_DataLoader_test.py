from f00131_create_DataLoader import *
def test_create_DataLoader():
    dataset = small_train_dataset
    dataloader = create_DataLoader(dataset, shuffle=True, batch_size=8)
    assert len(dataloader) == len(dataset) // 8

    dataset = small_eval_dataset
    dataloader = create_DataLoader(dataset, batch_size=8)
    assert len(dataloader) == len(dataset) // 8


test_create_DataLoader()
