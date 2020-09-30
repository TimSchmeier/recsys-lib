from recsyslib.modelmixin import ModelMixin
import numpy as np
import tensorflow as tf
import os


def test_mixin(tmpdir):
    nusers = 10
    nitems = 8
    logfile = os.path.join(str(tmpdir), "log.txt")

    class MixTst(ModelMixin, tf.keras.Model):
        def __init__(self, num_users, num_items, **kwargs):
            super().__init__(num_users, num_items, **kwargs)

    u = np.random.choice(range(nusers), 10, replace=False).astype(np.int16)
    i = np.random.choice(range(nitems), 10).astype(np.int16)
    x = (u, i)
    y = np.random.normal(size=10)

    dense = np.zeros((nusers, nitems))
    for ui, ij, yk in zip(u, i, y):
        dense[ui, ij] = yk

    mm = MixTst(nusers, nitems, latent_dim=2, logfile=logfile, name="Mixtst")
    sp = mm.build_sparse_matrix(x, y)
    d = tf.sparse.to_dense(sp)
    assert np.allclose(d.numpy(), dense)
