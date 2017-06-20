import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import process_matt

X, T = process_matt.get_data()
X, T = shuffle(X, T)
Xtrain = X[:-100]
Ttrain = T[:-100]
Xtest = X[-100:]
Ttest = T[-100:]


# random weights
w = np.random.randn(X.shape[1])
w1 = np.array(w)  # see why i copied w in gradient descent part
b = 0
# Y = expit(X.dot(w) + b)


def classification_rate(T, P):
    return np.mean(T == P)


'''
def calculate_cross_entropy(T, Y):
    E = 0
    for i in range(len(X)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E
'''


def calculate_cross_entropy(T, Y):
    return -np.mean(T * np.log(Y) + (1 - T) * np.log(1 - Y))


# To be clear, T is outcomes we observed (from data)
# Y is predicted outcomes based on current weights

train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    Ytrain = expit(Xtrain.dot(w) + b)
    Ytest = expit(Xtest.dot(w) + b)
    cost_train = calculate_cross_entropy(Ttrain, Ytrain)
    cost_test = calculate_cross_entropy(Ttest, Ytest)

    # gradient descent formula is different in different videos
    w -= learning_rate * Xtrain.T.dot(Ytrain - Ttrain)
    w1 += learning_rate * Xtrain.T.dot(Ttrain - Ytrain)
    assert(np.array_equal(w, w1))

    # this is new...
    b -= learning_rate * (Ytrain - Ttrain).sum()

    train_costs.append(cost_train)
    test_costs.append(cost_test)

    if (i % 1000 == 0):
        print('Entropy error:', cost_train, cost_test)

print('Final train classification rate:', classification_rate(Ttrain, np.round(Ytrain)))
print('Final test classification rate:', classification_rate(Ttest, np.round(Ytest)))

legend1, = plt.plot(train_costs, label='train costs')
legend2, = plt.plot(test_costs, label='test costs')
plt.legend([legend1, legend2])
plt.show()
