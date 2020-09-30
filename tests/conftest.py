import pytest
import numpy as np


@pytest.fixture
def uir_tuples(scope="module"):
    nusers = 20
    nitems = 20
    x = (np.arange(nusers), np.random.choice(range(nitems), nusers))
    y = np.random.uniform(low=0, high=5.0, size=nusers)
    return nusers, nitems, x, y
