import numpy as np

# softmax on vector
a = np.random.randn(5)
expa = np.exp(a)
answer = expa / expa.sum()
print(answer)
print(answer.sum())

# softmax on array
A = np.random.randn(100,5)
print(A)
expA = np.exp(A)
answer = expA / expA.sum(axis=1, keepdims=True)
print(answer)
print(answer.sum(axis=1))