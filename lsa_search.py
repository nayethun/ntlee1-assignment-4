# lsa_search.py

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds

# Load the dataset
def load_data():
    newsgroups = fetch_20newsgroups(subset='all')
    documents = newsgroups.data
    return documents

# Preprocess and compute LSA
def compute_lsa(documents, n_components=100):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    u, s, vt = svds(X.asfptype(), k=n_components)
    # Reverse the components
    s = s[::-1]
    u = u[:, ::-1]
    vt = vt[::-1, :]
    # Compute X_lsa
    X_lsa = u * s    # Shape: (n_documents, n_components)
    return X_lsa, vt, vectorizer

# Process user query
def process_query(query, vt, vectorizer):
    query_vec = vectorizer.transform([query])    # Shape: (1, n_terms)
    query_vec_dense = query_vec.toarray()        # Convert to dense array
    query_lsa = np.dot(query_vec_dense, vt.T)    # Shape: (1, n_components)
    query_lsa = query_lsa.flatten()              # Convert to 1-D array
    return query_lsa

# Compute cosine similarities
def compute_similarities(query_lsa, X_lsa):
    # X_lsa: (n_documents, n_components)
    # query_lsa: (n_components,)
    query_norm = np.linalg.norm(query_lsa)
    X_norms = np.linalg.norm(X_lsa, axis=1)
    dot_products = X_lsa.dot(query_lsa)
    similarities = dot_products / (X_norms * query_norm)
    return similarities

# Retrieve top documents
def get_top_documents(similarities, documents, top_n=5):
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_docs = [(documents[i], similarities[i]) for i in top_indices]
    return top_docs
