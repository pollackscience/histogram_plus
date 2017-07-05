"""
Bayesian Block implementation
=============================

Dynamic programming algorithm for finding the optimal adaptive-width histogram.

Based on Scargle et al 2012 [1]
Initial Python Implementation [2]
AstroML Implementation [3]


References
----------
.. [1] http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
.. [2] http://jakevdp.github.com/blog/2012/09/12/dynamic-programming-in-python/
.. [3] https://github.com/astroML/astroML/blob/master/astroML/density_estimation/bayesian_blocks.py
"""
from __future__ import division
import numpy as np


class Events(object):
    """Fitness for binned or unbinned events

    Parameters
    ----------
    p0 : float
        False alarm probability, used to compute the prior on N
        (see eq. 21 of Scargle 2012).  Default prior is for p0 = 0.
    gamma : float or None
        If specified, then use this gamma to compute the general prior form,
        p ~ gamma^N.  If gamma is specified, p0 is ignored.
    """
    def __init__(self, p0=0.05, gamma=None):
        self.p0 = p0
        self.gamma = gamma

    def validate_input(self, t, x):
        """Check that input is valid"""
        pass

    def fitness(self, N_k, T_k):
        # eq. 19 from Scargle 2012
        return N_k * (np.log(N_k) - np.log(T_k))

    def prior(self, N, Ntot):
        if self.gamma is not None:
            return self.gamma_prior(N, Ntot)
        else:
            # eq. 21 from Scargle 2012
            return 4 - np.log(73.53 * self.p0 * (N ** -0.478))

    def p0_prior(self, N, Ntot):
        # eq. 21 from Scargle 2012
        return 4 - np.log(73.53 * self.p0 * (N ** -0.478))

    def gamma_prior(self, N, Ntot):
        """Basic prior, parametrized by gamma (eq. 3 in Scargle 2012)"""
        if self.gamma == 1:
            return 0
        else:
            return (np.log(1 - self.gamma) - np.log(1 - self.gamma ** (Ntot + 1)) + N *
                    np.log(self.gamma))

    # the fitness_args property will return the list of arguments accepted by
    # the method fitness().  This allows more efficient computation below.
    @property
    def args(self):
        try:
            # Python 2
            return self.fitness.func_code.co_varnames[1:]
        except AttributeError:
            return self.fitness.__code__.co_varnames[1:]


def bayesian_blocks(t, x=None, p0=0.05, gamma=None):
    """Bayesian Blocks Implementation

    This is a flexible implementation of the Bayesian Blocks algorithm
    described in Scargle 2012 [1]_

    Parameters
    ----------
    t : array_like
        data times (one dimensional, length N)
    x : array_like (optional)
        data values

    Returns
    -------
    edges : ndarray
        array containing the (N+1) bin edges

    Examples
    --------
    Event data:

    >>> t = np.random.normal(size=100)
    >>> bins = bayesian_blocks(t, fitness='events', p0=0.01)

    Event data with repeats:

    >>> t = np.random.normal(size=100)
    >>> t[80:] = t[:20]
    >>> bins = bayesian_blocks(t, fitness='events', p0=0.01)

    References
    ----------
    .. [1] Scargle, J `et al.` (2012)
           http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    See Also
    --------
    astroML.plotting.hist : histogram plotting function which can make use
                            of bayesian blocks.
    """
    # validate array input
    t = np.asarray(t, dtype=float)
    if x is not None:
        x = np.asarray(x)

    # verify the fitness function
    # if x is not None and np.any(x % 1 > 0):
    #     raise ValueError("x must be integer counts for fitness='events'")
    fitfunc = Events(p0, gamma)

    # find unique values of t
    t = np.array(t, dtype=float)
    assert t.ndim == 1
    unq_t, unq_ind, unq_inv = np.unique(t, return_index=True,
                                        return_inverse=True)

    # if x is not specified, x will be counts at each time
    if x is None:
        if len(unq_t) == len(t):
            x = np.ones_like(t)
        else:
            x = np.bincount(unq_inv)

        t = unq_t

    # if x is specified, then we need to sort t and x together
    else:
        x = np.asarray(x)

        if len(t) != len(x):
            raise ValueError("Size of t and x does not match")

        if len(unq_t) != len(t):
            raise ValueError("Repeated values in t not supported when "
                             "x is specified")
        t = unq_t
        x = x[unq_ind]

    N = t.size
    # validate the input

    fitfunc.validate_input(t, x)

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    # arrays to store the best configuration
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    # -----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    # -----------------------------------------------------------------
    for R in range(N):
        # Compute fit_vec : fitness of putative last block (end at R)
        kwds = {}

        # T_k: width/duration of each block
        if 'T_k' in fitfunc.args:
            kwds['T_k'] = block_length[:R + 1] - block_length[R + 1]

        # N_k: number of elements in each block
        if 'N_k' in fitfunc.args:
            kwds['N_k'] = np.cumsum(x[:R + 1][::-1])[::-1]

        # evaluate fitness function
        fit_vec = fitfunc.fitness(**kwds)

        A_R = fit_vec - fitfunc.prior(R + 1, N)
        A_R[1:] += best[:R]

        i_max = np.argmax(A_R)
        last[R] = i_max
        best[R] = A_R[i_max]

    #-----------------------------------------------------------------
    # Now find changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]
