import theano.tensor as T
c = T.scalar('c')
v = T.vector('v')
A = T.matrix('A')
w = A.dot(v)

import theano
matrix_times_vector = theano.function(inputs=[A, v], outputs=w)

import numpy as np
A_val = np.array([[1, 2], [3, 4]])
v_val = np.array([5, 6])
w_val = matrix_times_vector(A_val, v_val)
print(w_val)

# normal vars aren't changeable therefore...
x = theano.shared(20.0, 'x')

# create arbitrary cost function that has a minimum
cost = x * x + x + 1

# theano calculates gradients directly
x_update = x - 0.3 * T.grad(cost, x)

# 1. no inputs below as this is just example, in future inputs will be
# data and labels
# 2. we are solving for x
train = theano.function(inputs=[], outputs=cost, updates=[(x, x_update)])
for i in range(25):
    # return value of train() is return of cost()
    # using current value of x inside train()
    cost_val = train()
    print(cost_val)

# print optimal value of x
print(x.get_value())
