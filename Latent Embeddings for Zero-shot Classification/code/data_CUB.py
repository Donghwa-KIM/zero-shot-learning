import scipy.io

data_CUB = scipy.io.loadmat('C:/Users/donghwa/PycharmProjects/personal_project/matfile/data_CUB.mat')
data_Y = scipy.io.loadmat('C:/Users/donghwa/PycharmProjects/personal_project/matfile/forYdata.mat')

# images embedding matrix(cnn feature obtained from gooLeNet) of train+validation set, each row is an image instance
trainval_X = data_CUB['trainval_X']
train_X = data_CUB['train_X']
val_X = data_CUB['val_X']
test_X = data_CUB['test_X']


## trainval_Y('word2vec') is the class embedding matrix of train+validation set

# train+validation
trainval_Y_cont = data_Y['trainval_Y_cont']
trainval_Y_glove = data_Y['trainval_Y_glove']
trainval_Y_word2vec = data_Y['trainval_Y_word2vec']
trainval_Y_wordnet = data_Y['trainval_Y_wordnet']
trainval_labels = data_CUB['trainval_labels']

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

