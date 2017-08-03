# argmaxOverMatrices(x, y, W)
# x:an image embedding instance
# y:a class embedding
# W:a cell array of embeddings
#  best_score: best bilinear score among all the embeddings
#  best_idx: index of the embedding with the best score

import numpy as np
'''
W = {0: np.array([[ 0.46579405,  0.39880095],
        [ 0.19330256,  0.49030222],
        [ 0.42094334,  0.54735289]]), 1: np.array([[ 0.07621766,  0.44164885],
        [ 0.15244866,  0.20302131],
        [ 0.48139197,  0.02113242]]), 2: np.array([[ 0.49874455,  0.38332187],
        [ 0.41360039,  0.25925641],
        [ 0.07110112,  0.35486235]]), 3: np.array([[ 0.5679922 ,  0.39967469],
        [ 0.43017881,  0.24201431],
        [ 0.27139092,  0.07525981]]), 4: np.array([[ 0.55957126,  0.51904194],
        [ 0.05055281,  0.57601613],
        [ 0.37441431,  0.25932433]]), 5: np.array([[ 0.06522629,  0.5026019 ],
        [ 0.45290199,  0.08549494],
        [ 0.53525909,  0.14037973]]), 6: np.array([[ 0.42957483,  0.5133507 ],
        [ 0.38488892,  0.51159427],
        [ 0.27712911,  0.09636296]]), 7: np.array([[ 0.08043121,  0.08000775],
        [ 0.46852862,  0.57296815],
        [ 0.49636085,  0.19417568]]), 8: np.array([[ 0.24230089,  0.52220915],
        [ 0.23523859,  0.51063579],
        [ 0.33301438,  0.3576675 ]]), 9: np.array([[ 0.52819669,  0.13094418],
        [ 0.41653341,  0.41234314],
        [ 0.15640626,  0.03220666]])}

y= np.array([5, 7])
x = np.array([4, 5, 6])
i=0
'''
def argmaxOverMatrices(x, y, W):
    K = len(W)
    # minimum value
    best_score = -1e12
    best_idx = -1
    score = np.zeros(K)

    for i in range(0,K):
        projected_x = np.dot(x, W[i])
        score[i] = np.dot(projected_x, y)
        if (score[i] > best_score):
            best_score = score[i]
            best_idx = i

    return (best_score,best_idx)