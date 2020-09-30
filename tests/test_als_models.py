from recsyslib.als import (
    ImplicitMF,
    LogisticMF,
    BVD,
    SVD,
)


def test_LogisticMF(uir_tuples):
    nusers, nitems, x, y = uir_tuples
    logMF = LogisticMF(nusers, nitems, latent_dim=2)
    logMF.fit(x, y, epochs=2)
    uembed = logMF.user_embeddings
    assert uembed.shape == (nusers, 2)
    iembed = logMF.item_embeddings
    assert iembed.shape == (nitems, 2)


def test_ImplicitMF(uir_tuples):
    nusers, nitems, x, y = uir_tuples
    iMF = ImplicitMF(nusers, nitems, latent_dim=2)
    iMF.fit(x, y, epochs=2)
    uembed = iMF.user_embeddings
    assert uembed.shape == (nusers, 2)
    iembed = iMF.item_embeddings
    assert iembed.shape == (nitems, 2)


def test_BVD(uir_tuples):
    nusers, nitems, x, y = uir_tuples
    bvd = BVD(nusers, nitems, num_rowclusters=2, num_columnclusters=3)
    bvd.fit(x, y, epochs=2)


def test_SVD(uir_tuples):
    nusers, nitems, x, y = uir_tuples
    svd = SVD(nusers, nitems, latent_dim=2)
    svd.fit(x, y)
    uembed = svd.user_embeddings
    assert uembed.shape == (nusers, 2)
    iembed = svd.item_embeddings
    assert iembed.shape == (nitems, 2)
