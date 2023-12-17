# requirements_file --------------------

!pip install -U sentence-transformers transformers numpy scikit-learn

# function_import --------------------

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans

# function_code --------------------

def cluster_customer_reviews(reviews, num_clusters=5):
    # Load the pretrained SentenceTransformer model
    model = SentenceTransformer('nikcheerla/nooks-amd-detection-v2-full')

    # Encode reviews into high-dimensional vector space
    embeddings = model.encode(reviews)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    # Return the cluster labels for each review
    return labels

# test_function_code --------------------

def test_cluster_customer_reviews():
    print("Testing cluster_customer_reviews function.")
    sample_reviews = [
        "Great service but the food was just average.",
        "Had a fantastic time here, the staff were amazing!",
        "Quite disappointed with the delay in service.",
        "The ambiance was nice but my order was prepared wrong.",
        "Delicious food, will come back for sure!"
    ]

    # Test case: Cluster sample reviews into 2 clusters
    print("Testing case grouping into 2 clusters.")
    labels_2_clusters = cluster_customer_reviews(sample_reviews, num_clusters=2)
    assert len(set(labels_2_clusters)) == 2, f"Test case failed: Expected 2 clusters, found {len(set(labels_2_clusters))}"

    print("Testing finished.")

# Run the test function
test_cluster_customer_reviews()