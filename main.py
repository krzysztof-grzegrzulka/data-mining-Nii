import numpy as np
import pandas as pd
from numpy import argmax
from scipy.stats import mode
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from tqdm import tqdm

from functions.text_tokenizer import text_tokenizer
from functions.top_x_documents import top_x_documents
from functions.top_x_tokens import top_x_tokens

csv_files = ['Fake', 'True']

for file in csv_files:
    csv_text = ""
    csv_list = []
    csv_location = 'dataset/' + file + '.csv'

    df = pd.read_csv(csv_location)
    for i in tqdm(range(len(df['title']))):
        csv_text += df['title'].iloc[i] + " "
        csv_list.append(df['title'].iloc[i])

    vectorizer_count = CountVectorizer(tokenizer=text_tokenizer)
    X_transform = vectorizer_count.fit_transform(csv_list)

    vectorizer_tfidf = TfidfVectorizer(tokenizer=text_tokenizer)
    tfidf_transform = vectorizer_tfidf.fit_transform(csv_list)

    print("top 10 most often occurring")
    print(top_x_tokens(X_transform.toarray().sum(axis=0),
                       vectorizer_count.get_feature_names_out(), 10))

    print("top 10 most important")
    print(top_x_tokens(tfidf_transform.toarray().sum(axis=0),
                       vectorizer_tfidf.get_feature_names_out(), 10))

    print("top 10 documents with highest number of tokens")
    print(top_x_documents(X_transform.toarray().sum(axis=1), 10))
