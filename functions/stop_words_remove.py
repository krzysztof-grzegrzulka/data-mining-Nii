from nltk.corpus import stopwords


def stop_words_remove(source_text: str) -> list:
    stop_words = set(stopwords.words('english'))
    list_of_words = source_text.split(' ')
    return [word for word in list_of_words if word not in stop_words]
