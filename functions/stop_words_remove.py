from nltk.corpus import stopwords


def stop_words_remove(source_text: list) -> list:
    stop_words = set(stopwords.words('english'))
    return [word for word in source_text if word not in stop_words]
