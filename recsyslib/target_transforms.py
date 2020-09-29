import numpy as np


def interact_to_confidence(val, alpha):
    return alpha * val


def loginteract_to_confidence(val, alpha, eps=1e-5):
    return 1.0 + alpha * np.log(1 + val / eps)


def scale_zero_one(y):
    y -= np.min(y)
    y /= np.max(y)
    return y
