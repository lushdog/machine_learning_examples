# same as 'ann_class/backprop_matt.py' but using tensorflow
# using tensorflow means:
# * no calculation of derivatives by hand
# * forward() does not return softmax (returns logits) as tf cost function takes logits
# * no manual gradient descent

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
import tensorflow as tf


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2
    # we do not return the output of the softmax when using
    # Tensorflow, these are called 'logits'


def main():

    Nclass = 500  # num observations of each class
    D = 2         # num dimensions
    M = 3         # 1 hidden layer with size of 3
    K = 3         # num classes

    # three gaussian clouds each w/ different centers
    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])

    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)
    X = np.vstack((X1, X2, X3))
    N = len(Y)    # num observations
    T = pre.OneHotEncoder(sparse=False).fit_transform(Y.reshape(-1, 1))

    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
    plt.show()

    # create placeholders
    # NOTE: X and T get pushed to these placeholders in session.run()
    # shape[None, D] means we can pass in any value for shape[0]
    tfx = tf.placeholder(tf.float32, [None, D])
    tfy = tf.placeholder(tf.float32, [None, K])

    W1 = init_weights([D, M])
    b1 = init_weights([M])
    W2 = init_weights([M, K])
    b2 = init_weights([K])

    py_x = forward(tfx, W1, b1, W2, b2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=tfy))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    for i in range(1000):
        session.run(train_op, feed_dict={tfx: X, tfy: T})
        pred = session.run(predict_op, feed_dict={tfx: X, tfy: T})
        if i % 10 == 0:
            print(np.mean(Y == pred))


if __name__ == '__main__':
    main()
