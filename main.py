import numpy as np
import pandas as pd
from numpy import argmax
from scipy.stats import mode
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from functions.text_tokenizer import text_tokenizer

csv_files = ['Fake', 'True']

for file in csv_files:
    csv_text = ""
    csv_list = []
    csv_location = 'dataset/' + file + '.csv'

    df = pd.read_csv(csv_location)
    for i in tqdm(range(len(df['title']))):
        csv_text += df['title'].iloc[i] + " "
        csv_list.append(df['title'].iloc[i])

    vectorizer = CountVectorizer(tokenizer=text_tokenizer(csv_text))
    # X_transform = vectorizer.fit_transform(csv_list[:3])
    X_transform = vectorizer.fit_transform(csv_list)
    print(vectorizer.get_feature_names_out())
    X_transform_array = X_transform.toarray()
    sum_columns = X_transform_array.sum(axis=0)
    # sum_columns.tolist().sort(reverse=True)
    # sum_columns.sort()
    sum_columns = sum_columns.tolist()
    sum_columns.sort(reverse=True)
    # print(sum_columns[-9:])
    print(sum_columns[:9])
    # print(type(sum_columns))
    # print(mode(X_transform_array, axis=0))
    # for X in X_transform_array:
    #     top10_mo = argmax(X_transform_array, axis=0)
    #     print(top10_mo)
