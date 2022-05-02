import pandas as pd
from sklearn import metrics
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, RandomForestClassifier)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from functions.text_tokenizer import text_tokenizer

df_true_news = pd.read_csv(r'dataset/True.csv')
df_fake_news = pd.read_csv(r'dataset/Fake.csv')

df_true_news['dataset'] = 1
df_fake_news['dataset'] = 0

df_true_fake = pd.concat([df_true_news, df_fake_news])

vectorizer_count = CountVectorizer(tokenizer=text_tokenizer)
X_transform = vectorizer_count.fit_transform(df_true_fake['title'])

x_train, x_test, y_train, y_test = train_test_split(
    X_transform, df_true_fake['dataset'], test_size=0.4,
    random_state=20)

classif_list = [AdaBoostClassifier(), BaggingClassifier(),
                DecisionTreeClassifier(), LinearSVC(),
                RandomForestClassifier()]

for classifier in classif_list:
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    print(f'{classifier}\'s accuracy: ',
          round(metrics.accuracy_score(y_test, prediction), 5))
