import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sklearn
from sklearn.decomposition import TruncatedSVD

import schedule
import time
import threading


def recommend():
    threading.Timer(900, recommend).start()
    df = pd.read_csv('./ratings_Beauty.csv')
    popular_products = pd.DataFrame(df.groupby('productRate')['rating'].count())
    most_popular = popular_products.sort_values('rating', ascending=False)
    Recommend_for_new_user = most_popular.head(10)
    print(Recommend_for_new_user)

    ratings_utility_matrix = df.head(1000).pivot_table(values='rating', index='buyerRate', columns='productRate', fill_value=0)

    X = ratings_utility_matrix.T

    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)

    correlation_matrix = np.corrcoef(decomposed_matrix)

    product_bought = "6117036094"

    product_names = list(X.index)
    product_ID = product_names.index(product_bought)

    correlation_product_ID = correlation_matrix[product_ID]

    Recommend_for_oldUser = list(X.index[correlation_product_ID > 0.90])
    Recommend_for_oldUser.remove(product_bought) 
    print(Recommend_for_oldUser[0:9])
    
recommend()