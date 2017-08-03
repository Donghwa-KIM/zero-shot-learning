import numpy as np
import numpy.matlib as nm
X = np.array([[1, 2, 3], [4, 5, 6], [2, 5, 3], [2, 1, 3], [2, 4, 5]])
Y = np.array([[1, 2, 3, 4, 5], [1, 4, 3, 6, 7]])
labels = np.array([1, 3, 5, 4, 2]).reshape(len([1, 3, 5, 4, 2]), 1)

def normalization(X, m, v, mx):
    n = X.shape[0]
    nargin=len(args)
    if (nargin == 1):
        X1 = X
        m = np.sum(X1,axis=0) / n
        X1 = X1 - nm.repmat(m, n, 1)
        v = np.sqrt(np.sum(np.power(X1,2) / n,axis=0))
        v[0] = 1
        X1 = X1 / nm.repmat(v, n, 1)
        mx = np.max(np.max(X1,axis=0))

Xn = (X - nm.repmat(m,n,1)) / nm.repmat(v, n, 1)
Xn = Xn / mx;

