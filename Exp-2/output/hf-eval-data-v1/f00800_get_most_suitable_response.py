from sentence_transformers import SentenceTransformer, util


def get_most_suitable_response(query: str, docs: list) -> str:
    """
    This function finds the most suitable response to a user question from a list of responses provided.

    Args:
        query (str): The user's question.
        docs (list): A list of possible responses.

    Returns:
        str: The most suitable response.
    """
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    return doc_score_pairs[0][0]