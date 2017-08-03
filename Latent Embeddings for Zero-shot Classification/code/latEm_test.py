import numpy as np
from scipy import sparse
X = np.array([[1,2,3],[4,5,6],[2,5,3],[2,1,3],[2,4,5]])
Y = np.array([[1, 2, 3, 4, 5],[1, 4, 3, 6, 7]])
labels = np.array([1,3,5,4,2]).reshape(len([1,3,5,4,2]),1)
n_train = X.shape[0]
n_class = Y.shape[1]
j=0

for i in range(0,K):
    W[i] = 1.0/np.sqrt(X.shape[1]) * np.random.rand(X.shape[1], Y.shape[0])

# W: latent embeddings
# X: images embedding matrix, each row is an image instance
# Y: class embedding matrix, each col is for a class
# labels: ground truth labels of all image instances


def latEm_test(W,X,Y,labels):

    all_scores = []
    n_samples = len(labels)
    n_class = len(np.unique(labels))

    K = len(W)
    scores = {}
    max_scores = np.zeros((K,n_samples))
    tmp_label = np.zeros((K,n_samples))

    for j in range(K):
        projected_X = np.dot(X , W[j])
        scores[j] = np.dot(projected_X, Y)
        # Maxima along the second axis
        # Maxima between classes per an image: col
        [max_scores[j,:], tmp_label[j,:]] = [np.amax(scores[j], axis = 1),np.argmax(scores[j],axis=1)+1]
    # Maxima between Ws: Weight
    [best_scores, best_idx] = [np.amax(max_scores, axis=0),np.argmax(max_scores,axis=0)]

    predict_label=np.zeros(n_samples)
    for k in range(n_samples):
        predict_label[k]=tmp_label[best_idx[k],k]


## compute the confusion matrix


# ground truth labels
label_mat = sparse.csr_matrix((np.repeat(1,n_class),(np.squeeze(labels.reshape(1,len(labels)))-1,np.arange(n_samples))),
    shape=(n_class,n_samples))

predict_mat = sparse.csr_matrix((np.repeat(1,n_class),(predict_label-1,np.arange(n_samples))),
    shape=(n_class,n_samples))

# predicted labels
conf_mat = sparse.csr_matrix.dot(label_mat,np.transpose(predict_mat))
conf_mat_diag = sparse.csr_matrix.diagonal(conf_mat)

# a kind of classes
n_per_class = np.squeeze(np.array(np.sum(sparse.csr_matrix.transpose(label_mat),0)))


# mean class accuracy
mean_class_accuracy = np.sum(conf_mat_diag / n_per_class) / n_class