import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

positive_reviews = BeautifulSoup(open('electronics/positive.review',
                                      encoding='utf-8')
                                 .read(),
                                 "html5lib")
positive_reviews = positive_reviews.findAll('review_text')
p_count = len(positive_reviews)
negative_reviews = BeautifulSoup(open('electronics/negative.review',
                                      encoding='utf-8')
                                 .read(),
                                 "html5lib")
negative_reviews = negative_reviews.findAll('review_text')
reviews = [pr.get_text() for pr in positive_reviews]
for nr in negative_reviews:
    reviews.append(nr.get_text())


# Custom tokenizer from:
# http://scikit-learn.org/stable/modules/feature_extraction.html
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
tfidf = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                        stop_words=stopwords,
                        strip_accents='unicode',
                        lowercase=True)
X = tfidf.fit_transform(reviews).todense()
Y = np.array([[1] if i < p_count else [0] for i in range(len(X))])

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y)
model = LogisticRegression()
model.fit(Xtrain, Ytrain.ravel())
print("Classification rate:", model.score(Xtest, Ytest.ravel()))

threshold = 0.5
for word, index in list(tfidf.vocabulary_.items()):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)
