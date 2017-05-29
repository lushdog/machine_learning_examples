# pylint: skip-file

import numpy as np
import matplotlib.pyplot as plt 

#make some fake data
N = 100
X = np.linspace(0, 6*np.pi, num=N)
Y = np.sin(X)

plt.scatter(X, Y)
plt.plot(X, Y)
plt.show()

# make data polynomial (add squared values to series)
def make_poly(X, deg):
    n = len(X)
    data = [np.ones(n)]
    for d in range(deg):
        data.append(X**(d+1))
    #stacked = np.vstack(data).T 
    stacked = np.stack(data, axis=0).T
    return stacked

def fit(X, Y):
    return np.linalg.solve(X.T.dot(X), (X.T).dot(Y))

def fit_and_display(X, Y, sample, deg):

    # select random samples to train
    N = len(X)
    train_idxs = np.random.choice(N, sample) # list of indexes
    Xtrain = X[train_idxs] 
    Ytrain = Y[train_idxs]

    foo = np.zeros((sample,2))
    foo[:,0] = Xtrain
    foo[:,1] = Ytrain
    print(foo.shape)
    print(foo)


    #show scatter of training data
    plt.scatter(Xtrain, Ytrain)
    plt.plot(X, Y)
    plt.show()

    # create polynomial of training sample and find line of best fit
    Xtrain_poly = make_poly(Xtrain, deg)
    w = fit(Xtrain_poly, Ytrain)

    # show results of fit
    X_poly = make_poly(X, deg)
    Y_hat = X_poly.dot(w)
    plt.plot(X, Y, color='b')
    plt.plot(X, Y_hat, color='y')
    plt.scatter(Xtrain, Ytrain, color='g')
    plt.title("deg = %d" % deg)
    plt.show()

for deg in (5, 6, 7, 8 , 9):
    fit_and_display(X, Y, 10, deg)


# mean squared error
def get_mse(Y, Yhat):
    d = Y - Yhat
    return d.dot(d) / len(d)



def plot_train_vs_test_curves():
    pass



