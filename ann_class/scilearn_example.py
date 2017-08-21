import sys
sys.path.append('../ann_logistic_extra')
import process_matt as process
from sklearn import neural_network
from sklearn import model_selection


X, Y = process.get_data()
nn = neural_network.MLPClassifier(hidden_layer_sizes=(30, 30), verbose=True, max_iter=10000, solver='adam', activation='relu')
Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(X, Y)
nn.fit(Xtrain, Ytrain)
train_score = nn.score(Xtrain, Ytrain)
test_score = nn.score(Xtest, Ytest)
print('train_score', train_score, 'test_score', test_score)
