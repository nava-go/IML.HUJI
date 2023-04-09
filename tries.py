import numpy as np


if __name__ == '__main__':
    X = np.full((2, 3), 1)
    y = np.full((2,), 1)
    y[0] = 2
    y[1] = 3
    print(X)
    print(y)
    #print(y.transpose()*X)
    print(y * X)
    #print(X*y)