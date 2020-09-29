from recsyslib.metrics import precision, recall, dcg, idcg, ndcg
import numpy as np


def test_precision():
    truth = [1, 2, 3, 4, 5]
    pred = [1, 3, 5, 7, 9, 11]
    expected = 3.0 / 6.0
    assert precision(truth, pred) == expected


def test_recall():
    truth = [1, 2, 3, 4, 5]
    pred = [1, 5, 10]
    expected = 2.0 / 5.0
    assert recall(truth, pred) == expected


def test_dcg():
    rel = [3, 2, 3, 1, 2]
    truth = [1, 2, 3, 5, 6]
    pred = [1, 2, 3, 4, 5, 6, 7, 8]
    dcg1 = dcg(truth, pred, rel)
    assert np.allclose(np.round(dcg1, 3), 9.899)
    # unrelevant documents at beginning
    pred2 = [8, 7, 1, 2, 3, 4, 5, 6]
    dcg2 = dcg(truth, pred2, rel)
    assert dcg2 < dcg1
    assert np.allclose(np.round(idcg(6, rel), 3), 10.302)
    assert np.allclose(np.round(ndcg(truth, pred, rel), 3), 0.961)
    # perfect score
    pred3 = [1, 3, 2, 6, 5]
    assert ndcg(truth, pred3, rel) == 1.0
    assert ndcg(truth, pred3) == 1.0
