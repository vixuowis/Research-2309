from f00543_print_num_videos import *
def test_print_num_videos():
    train_dataset = Ucf101(split='train')
    val_dataset = Ucf101(split='val')
    test_dataset = Ucf101(split='test')

    assert print_num_videos() == (train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)
