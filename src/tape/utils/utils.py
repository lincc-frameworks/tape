class IndexCallable:
    """Provide getitem syntax for functions

    >>> def inc(x):
    ...     return x + 1

    >>> I = IndexCallable(inc)
    >>> I[3]
    4
    """

    __slots__ = ("fn", "dirty", "ensemble")

    def __init__(self, fn, dirty, ensemble):
        self.fn = fn
        self.dirty = dirty  # propagate metadata
        self.ensemble = ensemble  # propagate ensemble metadata

    def __getitem__(self, key):
        result = self.fn(key)
        result.dirty = self.dirty  # propagate metadata
        result.ensemble = self.ensemble  # propagate ensemble metadata
        return result
