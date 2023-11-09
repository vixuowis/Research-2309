from transformers import LayoutXLMForQuestionAnswering


def extract_cash_flow_info(question: str, financial_document: str) -> str:
    """
    This function uses a pre-trained model from Hugging Face Transformers to extract information about cash flow from financial documents.
    
    Args:
        question (str): The question related to cash flow.
        financial_document (str): The financial document from which to extract the information.
    
    Returns:
        str: The answer to the question based on the information in the financial document.
    """
    # Load the pre-trained model
    model = LayoutXLMForQuestionAnswering.from_pretrained('fimu-docproc-research/CZ_DVQA_layoutxlm-base')
    
    # Generate the answer to the question based on the financial document
    answer = model.generate_answer(question, financial_document)
    
    return answer