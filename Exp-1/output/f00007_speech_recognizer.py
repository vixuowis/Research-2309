from typing import *
from typing import *
from typing import *
import torchaudio

def speech_recognizer(data):
    	pipeline = torchaudio.transforms.MFCC()
    	result = []
    	for d in data:
    	    waveform, sample_rate = torchaudio.load(d['audio'])
    	    mfcc = pipeline(waveform)
    	    text = transcribe(mfcc)
    	    result.append({'text': text})
    	return result
