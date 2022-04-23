from sklearn.feature_extraction.text import CountVectorizer

from functions.text_tokenizer import text_tokenizer


def top10_most_often(csv_text, csv_list):
    vectorizer = CountVectorizer(tokenizer=text_tokenizer(csv_text))
    X_transform = vectorizer.fit_transform(csv_list[:3])
    X_transform_array = X_transform.toarray()
