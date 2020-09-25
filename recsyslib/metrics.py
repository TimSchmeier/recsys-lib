import numpy as np


def precision(y, ypred):
    return len(set(y).intersection(set(ypred))) / len(ypred)


def recall(y, ypred):
    return len(set(y).intersection(set(ypred))) / len(y)


def dcg(y, ypred, yrel=None):
    # discounted cume gain, assumes binary relevance.
    if not yrel:
        yrel = np.ones_like(y)
    score = 0.0
    # sum over (relevant score) / log(list rank + 1)
    for rank, (yhati, yreli) in enumerate(zip(ypred, yrel), 1):
        # if predicted item is in relevant set
        if yhati in y:
            # add the relevance score adjusted for that rank
            score += yreli / np.log(rank + 1)
    return score


def idcg(k, yrel=None):
    # highest possible dcg @ k, assume all items ranked perfectly
    if not yrel:
        yrel = np.ones(k)
    score = 0.0
    for rank, yreli in enumerate(yrel, 1):
        score += yreli / np.log(rank + 1)
    return score


def ndcg(y, ypred, yrel=None):
    score = dcg(y, ypred, yrel)
    k = len(y)
    if yrel:
        score /= idcg(k, sorted(yrel, reverse=True))
    else:
        score /= idcg(k)
    return score
