import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# visualize the data
def visualize(label, title):
    words = ''
    for msg in data[data['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis('off')
    plt.show()


visualize('spam', 'Spam word count')
visualize('ham', 'Ham word count')
