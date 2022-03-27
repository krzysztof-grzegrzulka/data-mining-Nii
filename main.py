import pandas as pd
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
    X_transform = vectorizer.fit_transform(csv_list[:3])
    print(vectorizer.get_feature_names_out())
    print(X_transform.toarray())
