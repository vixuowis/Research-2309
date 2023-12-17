# requirements_file --------------------

!pip install -U transformers==4.27.0.dev0 pytorch==1.13.1 datasets==2.9.0 tokenizers==0.13.2

# function_import --------------------

from transformers import AutoModelForSpeechClassification, Wav2Vec2Processor
from datasets import load_dataset

# function_code --------------------

def identify_language_from_audio(audio_file_path):
    """
    Identifies the language spoken in an audio file.

    Parameters:
    audio_file_path (str): The path to the audio file to be analyzed.

    Returns:
    str: The identified language.
    """
    # Load the model and processor
    model = AutoModelForSpeechClassification.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    processor = Wav2Vec2Processor.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')

    # Load the audio file
    audio_input, _ = processor(audio_file_path, return_tensors='pt', sampling_rate=16000)

    # Make prediction
    with torch.no_grad():
        logits = model(**audio_input).logits

    # Identify the most probable language
    predicted_id = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_id)

# test_function_code --------------------

def test_identify_language_from_audio():
    print("Testing started.")
    dataset = load_dataset("fleurs", split='language_id')
    # Assuming there's a `file` column in the dataset with audio file paths
    audio_file_path = dataset[0]['file']

    # Test case
    print("Testing identification of language.")
    identified_language = identify_language_from_audio(audio_file_path)
    assert identified_language in dataset.features['language_ids'].names, f"Language identification failed: {identified_language}"
    print("Identification of language successful.")

# Run the test function
test_identify_language_from_audio()