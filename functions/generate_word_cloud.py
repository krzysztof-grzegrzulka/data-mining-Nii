import csv
import ssl

import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud

from functions.bag_of_words import bag_of_words
from functions.sanitize_text import sanitize_text
from functions.stemming import stemming
from functions.stop_words_remove import stop_words_remove

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')


def generate_word_cloud(words_list: list):
    # src_text = ' '.join(words_list)
    # sanitized = sanitize_text(src_text)
    # sanitized_list = sanitized.split(' ')
    # stemmed = stemming(sanitized_list)
    # stop_removed = stop_words_remove(stemmed)

    # bow = bag_of_words(stop_removed)
    bow = bag_of_words(words_list)
    # print(bow)

    wc = WordCloud()
    wc.generate_from_frequencies(bow)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud.png')
    plt.show()
