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


def generate_word_cloud(csv_filename: str):
    csv_content = []
    csv_location = 'dataset/' + csv_filename + '.csv'
    with open(csv_location) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                csv_content += row

    csv_text = ' '.join(csv_content)
    sanitized = sanitize_text(csv_text)
    stemmed = stemming(sanitized)
    stop_removed = stop_words_remove(stemmed)

    bow = bag_of_words(stop_removed)
    # print(bow)

    wc = WordCloud()
    wc.generate_from_frequencies(bow)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(csv_filename + '_wordcloud.png')
    plt.show()

    print('Word cloud for ' + csv_filename + ' was generated')
