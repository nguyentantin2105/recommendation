import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sklearn
from sklearn.decomposition import TruncatedSVD

import schedule
import time
import threading
import csv

def recommend(userid):
    threading.Timer(900, recommend).start()
    list_recommend = []
    df = pd.read_csv('./out.csv')
    df = df.fillna(0)
    popular_products = pd.DataFrame(df.groupby('productRate')['rate'].count())
    most_popular = popular_products.sort_values('rate', ascending=False)
    Recommend_for_new_user = most_popular.head(10)
    Recommend_for_new_user.to_csv('new_user.csv')
    
    df_new = pd.read_csv('./new_user.csv')
    list_recommend_to_new_user = list(df_new["productRate"].values)
    
    ratings_utility_matrix = df.pivot_table(values='rate', index='buyerRate', columns='productRate', fill_value=0)
    X = ratings_utility_matrix.T

    df1 = df.drop_duplicates('buyerRate')
    SVD = TruncatedSVD(n_components=4)
    decomposed_matrix = SVD.fit_transform(X)

    correlation_matrix = np.corrcoef(decomposed_matrix)

    df_buyer = df1[df1['buyerRate'] == userid]
    product_bought = df_buyer['lastOrder'].values
    
    if (product_bought != 0):
        product_names = list(X.index)
        product_index = product_names.index(product_bought)

        correlation_product_ID = correlation_matrix[product_index]


        Recommend_for_oldUser = list(X.index[correlation_product_ID > 0.45])
        Recommend_for_oldUser.remove(product_bought)
        list_recommend = Recommend_for_oldUser
    else:
        list_recommend =  list_recommend_to_new_user
    
    return list_recommend
    
# print(recommend('60dc8cff14cc353d219f6a5d'))