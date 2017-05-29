import numpy as np
import pandas as pd
from course_project_data_processing import get_data

X, Y = get_data()
D = X.shape[1]
W = np.random.randn(D)
print(W)