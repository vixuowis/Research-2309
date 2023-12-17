# requirements_file --------------------

!pip install -U torch espnet_model_zoo

# function_import --------------------

import soundfile
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_text_to_speech(text, output_file='output.wav'):
    """
    Convert the given Chinese text to speech using a pre-trained ESPnet model
    and save the speech audio to a WAV file.

    Parameters:
    - text: The Chinese text to be converted to speech.
    - output_file: The filename for the output WAV file.

    Returns:
    - None
    """
    # Load the pre-trained Chinese TTS model
    text2speech = Text2Speech.from_pretrained('espnet/kan-bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best')

    # Convert the text to speech
    speech = text2speech(text)["wav"]

    # Save the generated speech to an output file
    soundfile.write(output_file, speech.numpy(), text2speech.fs)

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing started.")

    # 测试用例：转换文本为语音并保存
    print("Testing text-to-speech conversion [1/1] started.")
    sample_text = '你好，世界'  # Example Chinese text
    output_file = 'test_output.wav'  # Output filename for the test

    # 调用函数进行测试
    convert_text_to_speech(sample_text, output_file)

    # 检查生成的文件是否存在
    import os
    assert os.path.isfile(output_file), f"Test case failed: {output_file} was not created."

    # 进一步的测试可以包括文件大小检查、内容检查等，这里省略
    print("Testing text-to-speech conversion finished.")

# 运行测试函数
test_convert_text_to_speech()