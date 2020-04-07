"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    K, d = mixture.mu.shape
    f = np.zeros((n,K))
    delta = (X != 0)
    logpost = np.zeros((n, K))

    for u in range(n):
        tiled_vector = np.tile(X[u, :], (K, 1))
        sse = ( delta[u, :] * (tiled_vector - mixture.mu) ** 2).sum(axis=1)
        Cu = delta[u, :].sum()
        f[u, :] = np.log(mixture.p) - 0.5 * Cu * np.log(2 * np.pi * mixture.var) - sse / (2 * mixture.var)
        logpost[u, :] = f[u, :] - logsumexp(f[u, :])

    loglikelihood = logsumexp(f, axis = 1).sum()
    post = np.exp(logpost)

    return post, loglikelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu[j] - X) ** 2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p)

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    loglikelihood = None
    post, new_loglikelihood = estep(X, mixture)
    while (loglikelihood is None or new_loglikelihood - loglikelihood > 1e-6 * abs(new_loglikelihood)):
        loglikelihood = new_loglikelihood
        mixture = mstep(X, post)
        post, new_loglikelihood = estep(X, mixture)


    return mixture, post, new_loglikelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
