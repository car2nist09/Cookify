# Cookify 🍳

A fridge-to-recipe recommendation engine built with Python and Flask.

## What it does
Enter the ingredients you have in your fridge and Cookify uses a **K-Nearest Neighbors ML model** to find the most similar recipes from a dataset of 5,000 recipes. Results are ranked by ingredient similarity percentage.

## How it works
- Each recipe is converted into a vector of 0s and 1s (ingredient present or not)
- User input is converted into the same vector format
- KNN finds the 10 recipes with the most similar ingredient patterns using cosine similarity
- The same core concept behind Spotify and Netflix recommendations

## Tech stack
- Python, Flask
- scikit-learn (KNN, MultiLabelBinarizer)
- pandas, numpy
- HTML, CSS

## Setup
1. Clone the repo
2. Run `pip install -r requirements.txt`
3. Download `RAW_recipes.csv` from [Food.com dataset on Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) and place in root folder
4. Run `python3 app.py`
5. Open `http://127.0.0.1:5000`