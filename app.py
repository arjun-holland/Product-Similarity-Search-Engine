# from flask import Flask, request, jsonify
# import json
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer

# # Load product data
# with open("products.json", "r") as f:
#     products = json.load(f)

# # Load model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Encode product descriptions
# product_descriptions = [p["description"] for p in products]
# embeddings = model.encode(product_descriptions)
# embeddings = np.array(embeddings, dtype='float32')

# # Create FAISS index (L2 distance)
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)

# # Initialize Flask app
# app = Flask(__name__)

# # Search endpoint
# @app.route("/search", methods=["GET"])
# def search():
#     query = request.args.get("query")
#     k = int(request.args.get("k", 3))  # default top 3
#     if not query:
#         return jsonify({"error": "Please provide a query"}), 400
    
#     # Encode query
#     query_vec = model.encode([query])
#     query_vec = np.array(query_vec, dtype='float32')
    
#     # Search
#     distances, indices = index.search(query_vec, k)
#     results = []
#     for i, idx in enumerate(indices[0]):
#         results.append({
#             "name": products[idx]["name"],
#             "description": products[idx]["description"],
#             "distance": float(distances[0][i])
#         })
    
#     return jsonify({"query": query, "results": results})

# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "Product Similarity Search API is running!"})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)


from flask import Flask, request, render_template
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load product data
with open("products.json", "r") as f:
    products = json.load(f)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode product descriptions
product_descriptions = [p["description"] for p in products]
embeddings = model.encode(product_descriptions)
embeddings = np.array(embeddings, dtype='float32')

# Create FAISS index (L2 distance)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Initialize Flask app
app = Flask(__name__)

def search_products(query, k=3):
    query_vec = model.encode([query])
    query_vec = np.array(query_vec, dtype='float32')
    distances, indices = index.search(query_vec, k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "name": products[idx]["name"],
            "description": products[idx]["description"],
            "distance": float(distances[0][i])
        })
    return results

@app.route("/")
def home():
    return render_template("home.html", products=products)

@app.route("/search", methods=["GET", "POST"])
def search():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form.get("query")
        if query:
            results = search_products(query, k=3)
    return render_template("search.html", query=query, results=results)

@app.route("/add", methods=["GET", "POST"])
def add_product():
    global index, embeddings, product_descriptions

    if request.method == "POST":
        name = request.form.get("name")
        description = request.form.get("description")

        if name and description:
            products.append({"name": name, "description": description})

            # Save to JSON
            with open("products.json", "w") as f:
                json.dump(products, f, indent=4)

            # Rebuild FAISS index
            product_descriptions = [p["description"] for p in products]
            embeddings = model.encode(product_descriptions)
            embeddings = np.array(embeddings, dtype='float32')
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

        return home()
    return render_template("add.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
