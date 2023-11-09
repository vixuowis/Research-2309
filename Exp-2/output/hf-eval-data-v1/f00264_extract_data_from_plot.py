from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image


def extract_data_from_plot(image_path):
    """
    This function extracts data tables from plots and charts using the Pix2StructForConditionalGeneration model from Hugging Face Transformers.
    
    Parameters:
    image_path (str): The path to the image file of the plot or chart.
    
    Returns:
    str: The extracted data table in a linearized format.
    """
    # Load the pre-trained model and processor
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    
    # Load the image
    image = Image.open(image_path)
    
    # Generate the inputs for the model
    inputs = processor(images=image, text='Generate underlying data table of the figure below:', return_tensors='pt')
    
    # Generate the predicted data table
    predictions = model.generate(**inputs, max_new_tokens=512)
    
    # Decode the predictions to get the data table
    data_table = processor.decode(predictions[0], skip_special_tokens=True)
    
    return data_table