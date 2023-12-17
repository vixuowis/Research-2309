# requirements_file --------------------

import subprocess

requirements = ["transformers", "torch"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

import subprocess

# 定义软件包列表
requirements = ['transformers', 'torch']

# 使用循环安装软件包
for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

# function_code --------------------

def analyze_emotion_from_speech(audio_file_path):
    """
    Analyzes the emotional content of a speech audio file.

    Args:
        audio_file_path (str): The file path of the audio file to analyze.

    Returns:
        str: The predicted emotion label of the speech audio file.

    Raises:
        FileNotFoundError: If the audio file does not exist at the specified path.
        ValueError: If the input is not a valid audio file.
    """
    # Emotion labels in the order of model outputs
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    # Load the pre-trained model and processor
    model = Wav2Vec2ForCTC.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    processor = Wav2Vec2Processor.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')

    # Load and process the audio file
    with open(audio_file_path, 'rb') as audio_file:
        input_values = processor(audio_file, return_tensors='pt', sampling_rate=16000).input_values

    # Get model predictions
    with torch.no_grad():
        logits = model(input_values).logits

    # Determine the predicted emotion
    predicted_indices = torch.argmax(logits, dim=-1)
    predicted_emotion = emotions[predicted_indices.item()]

    return predicted_emotion

# test_function_code --------------------

def test_analyze_emotion_from_speech():
    print("Testing started.")
    # Using a mock audio file path for testing
    test_audio_file_path = 'mock_audio_file.wav'

    # Test with the mock audio file
    print("Testing case [1/1] started.")
    predicted_emotion = analyze_emotion_from_speech(test_audio_file_path)
    # Since the function will be tested with a mock file or in a controlled environment,
    # the returned emotion can be pre-determined or the assert statement should be adapted.
    assert predicted_emotion == 'calm', f"Test case [1/1] failed: Expected 'calm', got {predicted_emotion}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_emotion_from_speech()