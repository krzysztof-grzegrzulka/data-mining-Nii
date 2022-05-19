import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, RandomForestClassifier)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize

from functions.bag_of_words import bag_of_words
from functions.sanitize_text import sanitize_text
from functions.stemming import stemming
from functions.stop_words_remove import stop_words_remove
from functions.text_tokenizer import text_tokenizer
from functions.generate_word_cloud import generate_word_cloud

from functions.top_x_documents import top_x_documents
from functions.top_x_tokens import top_x_tokens

# Dataset used: Twitter US Airline Sentiment
# link: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

# splitting one csv into multiple for easier operability
# file_name = 'dataset/airline_sentiment/Tweets.csv'
# source_before_split = pd.read_csv(file_name)
# for (airline_sentiment), \
#     group in source_before_split.groupby(['airline_sentiment']):
#     group.to_csv(f'dataset/airline_sentiment/{airline_sentiment}.csv',
#                  index=False)

df_all = pd.read_csv(r'dataset/airline_sentiment/Tweets.csv')
df_positive = pd.read_csv(r'dataset/airline_sentiment/positive.csv')
df_negative = pd.read_csv(r'dataset/airline_sentiment/negative.csv')
df_neutral = pd.read_csv(r'dataset/airline_sentiment/neutral.csv')
df_all_text = df_all['text']
df_positive_text = df_positive['text']
df_negative_text = df_negative['text']
df_neutral_text = df_neutral['text']

all_stemmed = []
positive_stemmed = []
negative_stemmed = []
neutral_stemmed = []

df_text_list = [df_all_text, df_positive_text, df_negative_text,
                df_neutral_text]
title_list = ['all tweets', 'positive tweets', 'negative tweets',
              'neutral tweets']
df_stemmed_list = [all_stemmed, positive_stemmed, negative_stemmed,
                   neutral_stemmed]

df_pos_neg = pd.concat([df_positive, df_negative])

print('Sentiment values')
print(df_all['airline_sentiment'].value_counts())
print('\n')

i = 0
print('Word Clouds')
for datafr in df_text_list:
    src_text = ' '.join(datafr)
    sanitized = sanitize_text(src_text)
    sanitized_list = sanitized.split()
    stop_removed = stop_words_remove(sanitized_list)
    stemmed = stemming(stop_removed)
    generate_word_cloud(stemmed, title_list[i])
    df_stemmed_list[i] = stemmed
    i += 1

print('TOP 10 stuff - positive')
vectorizer_count_positive = CountVectorizer(tokenizer=text_tokenizer)
X_transform_positive = vectorizer_count_positive. \
    fit_transform(df_positive['text'])

vectorizer_tfidf_positive = TfidfVectorizer(tokenizer=text_tokenizer)
tfidf_transform_positive = vectorizer_tfidf_positive. \
    fit_transform(df_positive['text'])

print("top 10 most often occurring - POSITIVE")
print(top_x_tokens(X_transform_positive.toarray().sum(axis=0),
                   vectorizer_count_positive.get_feature_names_out(), 10))

print("top 10 most important - POSITIVE")
print(top_x_tokens(tfidf_transform_positive.toarray().sum(axis=0),
                   vectorizer_count_positive.get_feature_names_out(), 10))

print("top 10 documents with highest number of tokens - POSITIVE")
print(top_x_documents(X_transform_positive.toarray().sum(axis=1), 10))

bow_positive = bag_of_words(positive_stemmed)
bow_negative = bag_of_words(negative_stemmed)

print('\nTOP 10 stuff - all')
vectorizer_count_all = CountVectorizer(tokenizer=text_tokenizer)
X_transform_all = vectorizer_count_all.fit_transform(df_all['text'])

vectorizer_tfidf_positive = TfidfVectorizer(tokenizer=text_tokenizer)
tfidf_transform_all = vectorizer_tfidf_positive.fit_transform(df_all['text'])

print("top 10 most often occurring - ALL")
print(top_x_tokens(X_transform_all.toarray().sum(axis=0),
                   vectorizer_count_all.get_feature_names_out(), 10))

print("top 10 most important - ALL")
print(top_x_tokens(tfidf_transform_all.toarray().sum(axis=0),
                   vectorizer_count_all.get_feature_names_out(), 10))

print("top 10 documents with highest number of tokens - ALL")
print(top_x_documents(tfidf_transform_all.toarray().sum(axis=1), 10))

# print("POSITIVE")
# generate_table_most_important(top_x_tokens(
#     X_transform_positive.toarray().sum(axis=0),
#     vectorizer_count_positive.get_feature_names_out(), 10),
#     bow_positive, 'count, positive')
# generate_plot_most_important(top_x_tokens(
#     X_transform_positive.toarray().sum(axis=0),
#     vectorizer_count_positive.get_feature_names_out(), 10),
#     bow_positive, 'count, positive')
#
# print("NEGATIVE")
# generate_table_most_important(top_x_tokens(
#     X_transform_negative.toarray().sum(axis=0),
#     vectorizer_count_negative.get_feature_names_out(), 10),
#     bow_negative, 'count, negative')
# generate_plot_most_important(top_x_tokens(
#     X_transform_negative.toarray().sum(axis=0),
#     vectorizer_count_negative.get_feature_names_out(), 10),
#     bow_negative, 'count, negative')

print('\nclassifiers')
x_train, x_test, y_train, y_test = train_test_split(
    X_transform_all, df_all['airline_sentiment'], test_size=0.4,
    random_state=20)

classif_list = [AdaBoostClassifier(), BaggingClassifier(),
                DecisionTreeClassifier(), LinearSVC()]

for classifier in classif_list:
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    print(f'{classifier}\'s accuracy: ',
          round(metrics.accuracy_score(y_test, prediction), 5))
    print(classification_report(y_test, prediction,
                                target_names=classifier.classes_))
    print('\n\n')

print('Classification report for RandomForestClassifier')
print('The most accurate one')
classif = RandomForestClassifier()
classif.fit(x_train, y_train)
fig, ax = plt.subplots(1, 1)
prediction = classif.predict(x_test)
print('RandomForestClassifier\'s accuracy: ',
      round(metrics.accuracy_score(y_test, prediction), 5))
conf_mat = confusion_matrix(y_test, prediction)
conf_mat = normalize(conf_mat, axis=0, norm='l1')
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                              display_labels=classif.classes_)
ax.title.set_text('RandomForestClassifier')
disp.plot(ax=ax)
plt.show()
print(classification_report(y_test, prediction, target_names=classif.classes_))
print(confusion_matrix(y_test, prediction))
