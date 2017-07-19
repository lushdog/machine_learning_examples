import numpy as np
import pandas as pd
import sklearn.preprocessing as pre

IS_MOBILE_IDX = 0
PRODUCTS_VIEWED_IDX = 1
VISIT_DURATION_IDX = 2
IS_RETURNING_IDX = 3
TIME_OF_DAY_IDX = 4
USER_ACTION_IDX = 5


def get_data():

    # load and prep data
    data = np.array(pd.read_csv(r'../ann_logistic_extra/ecommerce_data.csv'))
    X = np.array(data[:, :-1])
    Y = np.array(data[:, -1])
    X[:, [1, 2]] = pre.scale(X[:, [1, 2]])
    X = pre.OneHotEncoder(categorical_features=[TIME_OF_DAY_IDX],
                          sparse=False).fit_transform(X)
    return X, Y
