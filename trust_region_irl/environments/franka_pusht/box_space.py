import jax


class BoxSpace:
    """
    A limited continuous box space for the environment. It has a shape, lower bound and higher bound.
    """
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng):
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(self.dtype)
