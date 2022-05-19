import csv
import ssl

import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud

from functions.bag_of_words import bag_of_words

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')


def generate_word_cloud(words_list: list, title: str = 'word cloud'):
    bow = bag_of_words(words_list)

    wc = WordCloud()
    wc.generate_from_frequencies(bow)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.savefig('wordcloud.png')
    plt.show()
