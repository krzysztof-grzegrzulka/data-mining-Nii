from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def stop_words(source_text: str):
    text_tokens = word_tokenize(source_text)
    text_without_sw = ' '.join([word for word in text_tokens if not word in stopwords.words()])

    return text_without_sw
