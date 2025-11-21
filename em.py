from typing import Tuple
import numpy as np
from scipy.stats import multivariate_normal


def calculate_responsibilities(X: np.ndarray, means: np.ndarray, sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    :param X: data for clustering, shape: (N, D)
    :param means: means of K D-dimensional Gaussians, shape: (K, D)
    :param sigmas: covariance matrices for K D-dimensional Gaussians, shape: (K, D, D)
    :param weights: component weights, shape (K, )
    :return: responsibilities
    """
    N, K = X.shape[0], means.shape[0]
    responsibilities = np.zeros((N, K))

    for i in range(K):
        # Calculate unnormalized responsibility (numerator)
        # Adding a tiny regularization to sigma if needed to prevent crashes,
        # but keeping your logic as is:
        responsibilities[:, i] = weights[i] * multivariate_normal.pdf(X, mean=means[i], cov=sigmas[i])

    # Normalize (denominator)
    sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
    responsibilities = responsibilities / sum_responsibilities

    return responsibilities


def update_parameters(X: np.ndarray, means: np.ndarray, sigmas: np.ndarray,
                      weights: np.ndarray, responsibilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param X: data, shape: (N, D)
    :param means: current means, shape: (K, D)
    :param sigmas: current covariances, shape: (K, D, D)
    :param weights: current weights, shape (K, )
    :param responsibilities: shape (N, K)
    :return: means_new, sigmas_new, weights_new
    """
    N, K = X.shape[0], means.shape[0]

    means_new = np.zeros_like(means)
    sigmas_new = np.zeros_like(sigmas)
    weights_new = np.zeros_like(weights)

    # Effective number of points assigned to each cluster
    denom = responsibilities.sum(axis=0)

    # 1. Update Means
    # Transpose resp to (K, N) then dot with X (N, D) -> (K, D)
    means_new = (responsibilities.T @ X) / denom[:, np.newaxis]

    # 2. Update Covariances (Sigmas)
    for i in range(K):
        # Difference between data and the OLD mean (as per your provided code snippet)
        diff = X - means[i]

        # Weighted covariance calculation
        # Reshape resp column to (N, 1) for broadcasting
        weighted_diff = responsibilities[:, i, np.newaxis] * diff

        # (D, N) @ (N, D) -> (D, D)
        sigmas_new[i] = (weighted_diff.T @ diff) / denom[i]

    # 3. Update Weights
    weights_new = denom / N

    return means_new, sigmas_new, weights_new


def em(X: np.ndarray, K: int, max_iter: int, eps=1e-4, init_variance=1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the Expectation-Maximization algorithm.
    """
    assert max_iter > 0, 'max_iter must be a positive integer'
    N, D = X.shape[0], X.shape[1]

    # --- Initialization ---
    # Random means selected from data usually works better than random.random,
    # but keeping your initialization logic:
    means = np.random.random(size=(K, D))
    cov_mat = np.eye(D) * init_variance
    sigmas = np.repeat(cov_mat[None, :, :], K, axis=0)
    weights = 1. / K * np.ones((K,))

    print(f'Init: means={means}\n sigmas={sigmas}\n weights={weights}')

    log_likelihood = []
    soft_clusters = np.zeros((N, K))

    for it in range(max_iter):
        # --- E-Step ---
        responsibilities = calculate_responsibilities(X, means, sigmas, weights)

        # --- M-Step ---
        means, sigmas, weights = update_parameters(X, means, sigmas, weights, responsibilities)

        # --- Evaluation (Log-Likelihood) ---
        # Recalculate PDF with NEW parameters for likelihood check
        likelihood_matrix = np.zeros((N, K))
        for k in range(K):
            likelihood_matrix[:, k] = weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=sigmas[k])

        # Log-Sum-Exp approach for numerical stability is better, but using direct sum:
        total_likelihood = np.sum(likelihood_matrix, axis=1)
        current_ll = np.sum(np.log(total_likelihood + 1e-10))  # Added epsilon to avoid log(0)

        log_likelihood.append(current_ll)

        # Save soft clusters for return
        soft_clusters = likelihood_matrix / total_likelihood[:, np.newaxis]

        # Convergence Check
        if it > 1 and np.abs(log_likelihood[-1] - log_likelihood[-2]) < eps:
            print(f'Algorithm converged at iteration {it}.')
            break

    print(f'Fitted parameters: means={means}\n weights={weights}')
    return means, soft_clusters, np.array(log_likelihood)