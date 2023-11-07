from typing import *
from pytorchvideo.data import Ucf101

def print_num_videos():
    """Prints the number of videos in the train, validation, and test datasets."""
    train_dataset = Ucf101(split='train')
    val_dataset = Ucf101(split='val')
    test_dataset = Ucf101(split='test')

    print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)
