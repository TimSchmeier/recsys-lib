from recsyslib.searchutils import NNS, build_index
import numpy as np
from string import ascii_lowercase


def test_build_index():
    v = np.random.normal(size=(4, 10))
    t = build_index(v, num_trees=3)
    vn = t.get_item_vector(0)
    # norm check
    assert np.allclose(vn, v[0, :] / np.linalg.norm(v[0, :]))
    # index built and queryable
    assert len(t.get_nns_by_vector(vn, 2)) == 2


def test_NNS():
    e1 = np.random.normal(size=(5, 10))
    e2 = np.random.normal(size=(10, 10))
    t1 = build_index(e1, normalize=False, num_trees=3)
    t2 = build_index(e2, normalize=False, num_trees=3)
    nns = NNS()
    nns.add_index(t1, {kv: kv for kv in range(e1.shape[0])}, "e1")
    nns.add_index(
        t2, {l: i for i, l in enumerate(ascii_lowercase[: e2.shape[0]])}, "e2"
    )
    # check add index results
    assert len(nns.annoy_indices) == 2
    assert len(nns.entities) == 2
    # check methods
    assert nns.get_entity_index("e2", "d") == 3
    assert nns.get_entity_id("e1", 1) == 1
    assert np.allclose(nns.get_entity_vector("e2", "b"), e2[1, :].tolist())
    # only 5 nns in e1 index
    assert len(nns.query_by_vector("e1", e2[2, :], 10)) == 5
    assert len(nns.query_by_vector("e2", e2[2, :], 10)) == 10
    # check across entity query
    assert len(nns.get_similar("e2", "c", 5, "e1")) == 5
