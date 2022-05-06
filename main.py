import json

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, RandomForestClassifier)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from functions.bag_of_words import bag_of_words
from functions.generate_plot_most_important import generate_plot_most_important
from functions.generate_table_most_important import (
    generate_table_most_important)
from functions.sanitize_text import sanitize_text
from functions.stemming import stemming
from functions.stop_words_remove import stop_words_remove
from functions.text_tokenizer import text_tokenizer
from functions.generate_word_cloud import generate_word_cloud

# Dataset used - Goodreads Book Reviews
#   spoiler subset, raw
# Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley,
# "Fine-Grained Spoiler Detection from Large-Scale Review Corpora",
# in ACL'19
# https://arxiv.org/pdf/1905.13416.pdf
# https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/reviews

# cols = ['book_id', 'rating', 'review_text']
from functions.top_x_documents import top_x_documents
from functions.top_x_tokens import top_x_tokens

rows_limiter = 30000
cols = ['review_text']
data = []
reviews_list = []
df_text = ''
df_list = []
line_num = 0
file_name = 'dataset/goodreads_reviews_spoiler_raw.json'

with open(file_name) as f:
    for line in f:
        line_num += 1
        if line_num <= rows_limiter:
            doc = json.loads(line)
            # print(f'doc: {doc}; {type(doc)}')
            # lst = [doc['book_id'], doc['rating'], doc['review_text']]
            doc_str = ''.join(doc['review_text'])
            # print(f'doc_str: {doc_str}; {type(doc_str)}')
            # lst = [stop_removed]
            # lst2 = stop_removed
            lst = [doc['review_text']]
            lst2 = doc['review_text']
            data.append(lst)
            reviews_list.append(lst2)
        else:
            break

# print(reviews_list[0:2])

# sanitize text for later use #
src_text = ' '.join(reviews_list)
sanitized = sanitize_text(src_text)
sanitized_list = sanitized.split()
# stemmed = stemming(sanitized_list)
# stop_removed = stop_words_remove(stemmed)
stop_removed = stop_words_remove(sanitized_list)
stemmed = stemming(stop_removed)

# visualisations #
# generate_word_cloud(stop_removed)
generate_word_cloud(stemmed)

review_df = pd.DataFrame(data=data, columns=cols)
# print(review_df)

# TOP 10 #
for i in tqdm(range(len(review_df))):
    # df_text += review_df['review_text'].iloc[i] + ' '
    df_list.append(review_df['review_text'].iloc[i])

print('df_list generated')

vectorizer_count = CountVectorizer(tokenizer=text_tokenizer)
X_transform = vectorizer_count.fit_transform(df_list)

vectorizer_tfidf = TfidfVectorizer(tokenizer=text_tokenizer)
tfidf_transform = vectorizer_tfidf.fit_transform(df_list)

print('Count and Tfidf vectorizers generated')

print("top 10 most often occurring")
print(top_x_tokens(X_transform.toarray().sum(axis=0),
                   vectorizer_count.get_feature_names_out(), 10))

print("top 10 most important")
print(top_x_tokens(tfidf_transform.toarray().sum(axis=0),
                   vectorizer_tfidf.get_feature_names_out(), 10))
print("top 10 documents with highest number of tokens")
print(top_x_documents(X_transform.toarray().sum(axis=1), 10))

# TOP important table & plot #
bow_reviews = bag_of_words(stemmed)
print(bow_reviews)

generate_table_most_important(top_x_tokens(
    X_transform.toarray().sum(axis=0),
    vectorizer_count.get_feature_names_out(), 15),
    bow_reviews, 'count, book reviews')
generate_plot_most_important(top_x_tokens(
    X_transform.toarray().sum(axis=0),
    vectorizer_count.get_feature_names_out(), 15),
    bow_reviews, 'count, book reviews')

# classifiers #
review_df1 = review_df['review_text'][:14999]
review_df2 = review_df['review_text'][15000:]
review_df1['dataset'] = 1
review_df2['dataset'] = 0

review_df_1_2 = pd.concat([review_df1, review_df2])

vectorizer_count_class = CountVectorizer(tokenizer=text_tokenizer)
X_transform_class = vectorizer_count_class.\
    fit_transform(review_df_1_2['review_text'])

x_train, x_test, y_train, y_test = train_test_split(
    X_transform_class, review_df_1_2, test_size=0.2,
    random_state=20)

print('train_test_split finished')
classif_list = [AdaBoostClassifier(), BaggingClassifier(),
                DecisionTreeClassifier(), LinearSVC(),
                RandomForestClassifier()]

for classifier in classif_list:
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    print(f'{classifier}\'s accuracy: ',
          round(metrics.accuracy_score(y_test, prediction), 5))
