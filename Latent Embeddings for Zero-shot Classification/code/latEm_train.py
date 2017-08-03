
import scipy.io

data_CUB = scipy.io.loadmat('C:/Users/donghwa/Downloads/latEm/data_CUB.mat')

# images embedding matrix(cnn feature obtained from gooLeNet) of train+validation set, each row is an image instance
trainval_X = data_CUB['trainval_X']
train_X = data_CUB['train_X']
val_X = data_CUB['val_X']
test_X = data_CUB['test_X']

# trainval_Y('word2vec') is the class embedding matrix of train+validation set
# train
train_Y_cont = data_Y['train_Y_cont']
train_Y_glove = data_Y['train_Y_glove']
train_Y_word2vec = data_Y['train_Y_word2vec']
train_Y_wordnet = data_Y['train_Y_wordnet']
train_labels = data_CUB['train_labels']

# validation
val_Y_cont = data_Y['val_Y_cont']
val_Y_glove = data_Y['val_Y_glove']
val_Y_word2vec = data_Y['val_Y_word2vec']
val_Y_wordnet = data_Y['val_Y_wordnet']
val_labels = data_CUB['val_labels']

# test
test_Y_cont = data_Y['test_Y_cont']
test_Y_glove = data_Y['test_Y_glove']
test_Y_word2vec = data_Y['test_Y_word2vec']
test_labels = data_CUB['test_labels']


# ground truth labels
train_labels = data_CUB['train_labels']
val_labels = data_CUB['val_labels']
test_labels = data_CUB['test_labels']

from argmaxOverMatrices import argmaxOverMatrices
import numpy as np

# Temporal data
X = np.array([[1,2,3],[4,5,6],[2,5,3]])
Y = np.array([[1, 2, 3, 4, 5],[1, 4, 3, 6, 7]])
labels=np.array([1,3,5]).reshape(3,1)
n_train = X.shape[0]
n_class = Y.shape[1]

# Initialization
W = {}
K = 10
for i in range(0,K):
    W[i] = 1.0/np.sqrt(X.shape[1]) * np.random.rand(X.shape[1], Y.shape[0])
n_epoch = 10
i=0
eta=0.01


# SGD optimization for LatEm
for e in range(0,n_epoch):
    perm = np.random.permutation(n_train)
    for i in range(1,n_train):
        # A random image from a row
        ni = perm[i]
        best_i = -1
        # Allocate the ground truths to picked_y
        picked_y = labels[ni]
        # If they're same
        while(picked_y==labels[ni]):
            # Randomly generate again until those are different
            picked_y = np.random.randint(n_class)
        # If those are different
        # Random labeling
        [max_score, best_j] = argmaxOverMatrices(x=X[ni,:], y=Y[:,picked_y], W=W)
        # Grounded truth labeling
        [best_score_yi, best_j_yi] = argmaxOverMatrices(X[ni,:], Y[:,labels[ni]-1], W)
        if(max_score + 1 > best_score_yi):
            if(best_j==best_j_yi):
                W[best_j] = W[best_j] - eta * np.dot(X[ni,:].reshape(len(X[ni,:]),1),(Y[:,picked_y] - Y[:,labels[ni]-1].T))
            else
                W[best_j] = W[best_j] - eta * np.dot(np.asmatrix(X[ni,:]).T , np.asmatrix(Y[:,picked_y]))
                W[best_j_yi] = W[best_j_yi] + eta * np.dot(np.asmatrix(X[ni,:]).T , np.asmatrix(Y[:,labels[ni]]).T)