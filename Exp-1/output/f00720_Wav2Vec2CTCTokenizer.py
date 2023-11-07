from typing import *
from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer(vocab_file="my_vocab_file.txt")
