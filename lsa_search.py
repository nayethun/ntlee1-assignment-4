import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds

def load_data():
    newsgroups = fetch_20newsgroups(subset='all')
    documents = newsgroups.data
    return documents

def compute_lsa(documents, n_components=100):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    u, s, vt = svds(X.asfptype(), k=n_components)
    s = s[::-1]
    u = u[:, ::-1]
    vt = vt[::-1, :]
    X_lsa = u * s  
    return X_lsa, vt, vectorizer

def process_query(query, vt, vectorizer):
    query_vec = vectorizer.transform([query])    
    query_vec_dense = query_vec.toarray()       
    query_lsa = np.dot(query_vec_dense, vt.T)    
    query_lsa = query_lsa.flatten()              
    return query_lsa

def compute_similarities(query_lsa, X_lsa):
    query_norm = np.linalg.norm(query_lsa)
    X_norms = np.linalg.norm(X_lsa, axis=1)
    dot_products = X_lsa.dot(query_lsa)
    similarities = dot_products / (X_norms * query_norm)
    return similarities

def get_top_documents(similarities, documents, top_n=5):
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_docs = [(documents[i], similarities[i]) for i in top_indices]
    return top_docs
