from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# Function to cluster customer reviews based on their content similarity
# This function uses the SentenceTransformer model from Hugging Face Transformers to encode the reviews into a high-dimensional vector space
# The obtained embeddings are then clustered using the KMeans algorithm from sklearn
# The function returns the cluster labels for each review

def cluster_customer_reviews(reviews):
    # Load the pretrained SentenceTransformer model
    model = SentenceTransformer('nikcheerla/nooks-amd-detection-v2-full')
    # Encode the reviews into a high-dimensional vector space
    embeddings = model.encode(reviews)
    # Perform KMeans clustering on the embeddings
    kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings)
    # Return the cluster labels
    return kmeans.labels_