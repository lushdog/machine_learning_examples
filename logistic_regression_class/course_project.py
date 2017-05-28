import numpy as np
import pandas as pd

IS_MOBILE_IDX = 0
PRODUCTS_VIEWED_IDX = 1
VISIT_DURATION_IDX = 2
IS_RETURNING_IDX = 3
TIME_OF_DAY_IDX = 4
USER_ACTION_IDX = 5

# load and prep data
data = np.array(pd.read_csv(r'../ann_logistic_extra/ecommerce_data.csv'))
X = np.array(data[:, :-1])
Y = np.array(data[:, -1])
# ones = np.ones((len(X), 1))
# Xb = np.concatenate((X, ones), axis=1)
# print(Xb)
# print(Y)


# normalize numerical columns
X[:, 1] = (X[:, 1] - X[:, 1].mean() / X[:, 1].std())
X[:, 2] = (X[:, 2] - X[:, 2].mean() / X[:, 2].std())


# one-hot encoding for TIME_OF_DAY
zeros = np.zeros((len(X), 3))
X = np.concatenate((X, zeros), axis=1)
for row in X:
    time_of_day = row[TIME_OF_DAY_IDX]
    if time_of_day > 0:
        row[TIME_OF_DAY_IDX] = 0
        row[TIME_OF_DAY_IDX + int(time_of_day)] = 1

# only doing 2 classes (0, 1) in this project
# (binary logistic regression)
X1 = X[Y <= 1]
Y1 = Y[Y <= 1]

print(X1)
print(Y1)
