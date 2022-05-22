import pandas as pd
from sklearn import metrics
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, RandomForestClassifier)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from functions.sanitize_text import sanitize_text
from functions.stemming import stemming
from functions.stop_words_remove import stop_words_remove
from functions.text_tokenizer import text_tokenizer
from functions.generate_word_cloud import generate_word_cloud

col_list = ['rating', 'verified_reviews']
df_alexa = pd.read_csv(r'dataset/alexa_reviews.csv', sep=';',
                       encoding='cp1252', usecols=col_list)


print('Ratings')
print(df_alexa['rating'].value_counts())
print('\n')

src_text = ' '.join(df_alexa['verified_reviews'])
sanitized = sanitize_text(src_text)
sanitized_list = sanitized.split()
stop_removed = stop_words_remove(sanitized_list)
stemmed = stemming(stop_removed)
generate_word_cloud(stemmed, 'Alexa_reviews')

vectorizer_count = CountVectorizer(tokenizer=text_tokenizer)
X_transform = vectorizer_count. \
    fit_transform(df_alexa['verified_reviews'])

print('\nClassifiers')
x_train, x_test, y_train, y_test = train_test_split(
    X_transform, df_alexa['rating'], test_size=0.4,
    random_state=3)

classif_list = [AdaBoostClassifier(), BaggingClassifier(),
                DecisionTreeClassifier(), LinearSVC(dual=False),
                RandomForestClassifier()]

for classifier in classif_list:
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    print(f'{classifier}\'s accuracy: ',
          round(metrics.accuracy_score(y_test, prediction), 5))
    print(classification_report(y_test, prediction,
                                target_names=['1', '2', '3', '4', '5']))
    print('\n\n')
