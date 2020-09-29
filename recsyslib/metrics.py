import numpy as np


def precision(y, ypred):
    """
    Proportion of correct items predicted.

    Args:
        y (list): true items
        ypred (list): predicted items
    """
    return len(set(y).intersection(set(ypred))) / len(ypred)


def recall(y, ypred):
    """
    Proportion of relevent items found out of all relevant items.

    Args:
        y (list): true items
        ypred (list): predicted items
    """
    return len(set(y).intersection(set(ypred))) / len(y)


def dcg(y, ypred, yrel=None):
    """
    Discounted cumulative gain.

    Args:
        y (list): true items
        ypred (list): predicted items
        yrel (list): relevance scores for y
    """

    # discounted cume gain, assumes binary relevance.
    if not yrel:
        yrel = np.ones_like(y)
    score = 0.0
    # sum over (relevant score) / log(list rank + 1)
    for rank, yhati in enumerate(ypred, 1):
        # if predicted item is in relevant set
        if yhati in y:
            # find associated score with that item
            yreli = yrel[y.index(yhati)]
            # add the relevance score adjusted for that rank
            discount = np.log(rank + 1)
            s = yreli / discount
            score += s
    return score


def idcg(k, yrel=None):
    """
    Ideal discounted cumulative gain, assumes perfect ranking by yrel.

    Args:
        k (int): length of rankings
        yrel (list): relevance scores for y
    """
    # highest possible dcg @ k, assume all items ranked perfectly
    if not yrel:
        yrel = np.ones(k)
    # largest scores ranked first
    yrel = sorted(yrel, reverse=True)
    score = 0.0
    for rank, yreli in enumerate(yrel, 1):
        score += yreli / np.log(rank + 1)
    return score


def ndcg(y, ypred, yrel=None):
    """
    Normalized cumulative gain, corrects for different lengths of ranked lists.

    Args:
        y (list): true items
        ypred (list): predicted items
        yrel (list): relevance scores for y
    """

    score = dcg(y, ypred, yrel)
    k = len(y)
    score /= idcg(k, yrel)
    return score
