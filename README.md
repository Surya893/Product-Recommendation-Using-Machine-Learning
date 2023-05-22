# Product-Recommendation-Using-Machine-Learning

For this Project, I used Jupyter Notebook, where i developed the code. I imported a dataset from Kaggle containing files of different products available on Amazon onto my desktop, and then onto jupyter after that.

# How to Set Up Code for this:

Download Jupyter Notebook using the link below:

https://jupyter.org/install

After downloading, set up Jupyter, and proceed with the code provided:

```

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = pd.read_csv('/path/to/your/dataset.csv')

# Define input and target variables
X = data['description']
y = data['product_code']  # Replace 'product_id' with the correct column name

# Preprocess the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Compute similarity matrix
similarity_matrix = cosine_similarity(X)

# Define function to recommend products based on similarity
def recommend_products(product_code):
    idx = y[y == product_code].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_products = [y[i[0]] for i in similarity_scores[1:6]]
    return top_similar_products

# Test the recommendation function
recommend_products('P1234')

```
