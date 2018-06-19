import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud

data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
data.drop(data.columns[[2, 3, 4]], axis=1, inplace=True)
data.columns = ['labels', 'data']
data['b_labels'] = data['labels'].map({'ham': 0, 'spam': 1})
Y = data['b_labels'].as_matrix()
X = CountVectorizer(decode_error='ignore').fit_transform(data['data'])
# X = data.iloc[:, 1].values
# Y = data.iloc[:, 0].values
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y)

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("NB score:", model.score(Xtest, Ytest))

model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Ada score:", model.score(Xtest, Ytest))
