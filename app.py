from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
import ast

app = Flask(__name__)

print("Loading recipes...")
df = pd.read_csv('RAW_recipes.csv')
df = df[['name', 'ingredients']].dropna()
df = df.head(5000)

def parse_ingredients(ing_str):
    try:
        return ast.literal_eval(ing_str)
    except:
        return []

df['ingredients_list'] = df['ingredients'].apply(parse_ingredients)

mlb = MultiLabelBinarizer()
ingredient_matrix = mlb.fit_transform(df['ingredients_list'])

print("Training KNN model...")
knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(ingredient_matrix)
print("Ready!")

def find_recipes_knn(user_ingredients):
    user_ingredients = [i.strip().lower() for i in user_ingredients]
    known_ingredients = [i for i in user_ingredients if i in mlb.classes_]
    unknown_ingredients = [i for i in user_ingredients if i not in mlb.classes_]
    if not known_ingredients:
        return [], unknown_ingredients
    user_vector = mlb.transform([known_ingredients])
    if user_vector.sum() == 0:
        return [], unknown_ingredients
    distances, indices = knn.kneighbors(user_vector)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = round((1 - dist) * 100, 1)
        if similarity > 0:
            results.append({
                'name': df.iloc[idx]['name'],
                'ingredients': df.iloc[idx]['ingredients'],
                'similarity': similarity
            })
    return results, unknown_ingredients

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Recipies.html', methods=['GET', 'POST'])
def recipes():
    results = []
    user_input = ''
    unknown = []
    if request.method == 'POST':
        user_input = request.form['ingredients']
        results, unknown = find_recipes_knn(user_input.split(','))
    return render_template('Recipies.html', recipes=results, user_input=user_input, unknown=unknown)

if __name__ == '__main__':
    app.run(debug=True)