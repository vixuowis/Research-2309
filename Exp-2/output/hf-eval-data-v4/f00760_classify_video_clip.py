# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoImageProcessor, TimesformerForVideoClassification
import torch

# function_code --------------------

def classify_video_clip(video_data):
    # Load the pre-trained model and processor
    processor = AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k400')
    model = TimesformerForVideoClassification.from_pretrained('facebook/timesformer-base-finetuned-k400')
    
    # Process the video data
    inputs = processor(video_data, return_tensors='pt')
    
    # Perform inference with the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get the predicted class index and label
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_label = model.config.id2label[predicted_class_idx]
    return predicted_class_label

# test_function_code --------------------

def test_classify_video_clip():
    print("Testing started.")
    # Replace the below line to load your video dataset or a sample video clip
    video_data = list(np.random.randn(8, 3, 224, 224))  # mock video data for testing purposes  

    # Testing case: Check if the function returns a non-empty string
    print("Testing case [1/1] started.")
    predicted_label = classify_video_clip(video_data)
    assert isinstance(predicted_label, str) and len(predicted_label) > 0, f"Test case failed: Expected a non-empty string, got {predicted_label}"
    print("Testing finished.")

test_classify_video_clip()