import ssl

import nltk

from functions.stemming import stemming
from functions.stop_words_remove import stop_words_remove
from functions.sanitize_text import sanitize_text

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')


def text_tokenizer(src_text: str) -> list:
    tokenized_text_list = []
    text = sanitize_text(src_text)
    text_list = text.split(" ")
    text_list = stemming(text_list)
    text_list = stop_words_remove(text_list)

    for word in text_list:
        if len(word) > 3:
            tokenized_text_list.append(word)

    return tokenized_text_list
