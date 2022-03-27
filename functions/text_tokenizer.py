import ssl

import nltk

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


# def text_tokenizer(csv_filename: str):
def text_tokenizer(csv_text: str):
    output_words = []
    # csv_content = []
    # csv_text = ""
    # csv_location = 'dataset/' + csv_filename + '.csv'
    #
    # df = pd.read_csv(csv_location)
    # for i in tqdm(range(len(df['title']))):
    #     csv_text += df['title'].iloc[i] + " "
    # with open(csv_location) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         if line_count == 0:
    #             line_count += 1
    #         else:
    #             csv_content += row

    # csv_text = ' '.join(csv_content)
    sanitized = sanitize_text(csv_text)
    stemmed = stemming(sanitized)
    stop_removed = stop_words_remove(stemmed)

    for word in stop_removed:
        if len(word) > 3:
            output_words.append(word)
