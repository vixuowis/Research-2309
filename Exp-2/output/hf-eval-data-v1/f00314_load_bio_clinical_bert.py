from transformers import AutoTokenizer, AutoModel


def load_bio_clinical_bert():
    """
    This function loads the pre-trained model 'emilyalsentzer/Bio_ClinicalBERT' which is specifically trained on medical data.
    It can be used for various NLP tasks in the clinical domain, such as Named Entity Recognition (NER) and Natural Language Inference (NLI).
    """
    # Import the AutoTokenizer and AutoModel classes from the transformers library
    # Load the pre-trained model 'emilyalsentzer/Bio_ClinicalBERT'
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    return tokenizer, model