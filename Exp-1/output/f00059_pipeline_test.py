from f00059_pipeline import *
transcriber = pipeline(model='openai/whisper-large-v2', return_timestamps=True)
transcriber('https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac')
