from flask import Flask, render_template, request, jsonify
from lsa_search import load_data, compute_lsa, process_query, compute_similarities, get_top_documents

app = Flask(__name__)

# Load and process data at startup
print("Loading data and computing LSA...")
documents = load_data()
X_lsa, vt, vectorizer = compute_lsa(documents)
print("Data loaded and LSA computed.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    if not query:
        return jsonify({'results': []})

    print(f"Received query: '{query}'")

    try:
        query_lsa = process_query(query, vt, vectorizer)
        if query_lsa is None:
            print("No valid terms in query. Returning no results.")
            return jsonify({'results': []})
        similarities = compute_similarities(query_lsa, X_lsa)
        top_docs = get_top_documents(similarities, documents)
        results = [{'doc': doc, 'similarity': float(sim)} for doc, sim in top_docs]
        return jsonify(results=results)
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({'results': []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
