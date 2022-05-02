import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from functions.sanitize_text import sanitize_text
from functions.stop_words_remove import stop_words_remove
from functions.stemming import stemming
from functions.bag_of_words import bag_of_words
from functions.text_tokenizer import text_tokenizer
from functions.generate_table_most_important import (
    generate_table_most_important)
from functions.generate_plot_most_important import (
    generate_plot_most_important)
from functions.top_x_tokens import top_x_tokens

string_true_news = ""
string_fake_news = ""

df_true_news = pd.read_csv(r'dataset/True.csv')
df_fake_news = pd.read_csv(r'dataset/Fake.csv')

for i in tqdm(range(len(df_true_news['title']))):
    string_true_news += df_true_news['title'].iloc[i] + ' '

for i in tqdm(range(len(df_fake_news['title']))):
    string_fake_news += df_fake_news['title'].iloc[i] + ' '

stemmed_text_true_news = stemming(stop_words_remove(
    sanitize_text(string_true_news).split()))
stemmed_text_fake_news = stemming(stop_words_remove(
    sanitize_text(string_fake_news).split()))

bow_true_news = bag_of_words(stemmed_text_true_news)
bow_fake_news = bag_of_words(stemmed_text_fake_news)

vectorizer_count_true_news = CountVectorizer(tokenizer=text_tokenizer)
X_transform_count_true_news = vectorizer_count_true_news.fit_transform(
    df_true_news['title'])
vectorizer_count_fake_news = CountVectorizer(tokenizer=text_tokenizer)
X_transform_count_fake_news = vectorizer_count_fake_news.fit_transform(
    df_fake_news['title'])

print("TRUE NEWS")
generate_table_most_important(top_x_tokens(
    X_transform_count_true_news.toarray().sum(axis=0),
    vectorizer_count_true_news.get_feature_names_out(), 15),
    bow_true_news, 'count, true news')
generate_plot_most_important(top_x_tokens(
    X_transform_count_true_news.toarray().sum(axis=0),
    vectorizer_count_true_news.get_feature_names_out(), 15),
    bow_true_news, 'count, true news')

print("FAKE NEWS")
generate_table_most_important(top_x_tokens(
    X_transform_count_fake_news.toarray().sum(axis=0),
    vectorizer_count_fake_news.get_feature_names_out(), 15),
    bow_fake_news, 'count, fake news')
generate_plot_most_important(top_x_tokens(
    X_transform_count_fake_news.toarray().sum(axis=0),
    vectorizer_count_fake_news.get_feature_names_out(), 15),
    bow_fake_news, 'count, fake news')
