from f00470_transcriber import *
audio_file = "path/to/audio.wav"
result = transcriber(audio_file)
assert isinstance(result, dict)
assert "text" in result
