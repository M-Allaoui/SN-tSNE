# SN_tSNE.py
# Stand-alone semi-non-negative t-SNE (PNTSNE) with exact gradients
# Removes sklearn private imports and reimplements perplexity search.

import warnings
from time import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import linalg

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

MACHINE_EPS = 1e-12


# ------------------------ Utils: Perplexity & P ------------------------ #

def _hbeta(D_row, beta):
    """Given a row of squared distances D_row and precision beta = 1/(2*sigma^2),
    return conditional probs and Shannon entropy (in natural log)."""
    P = np.exp(-D_row * beta)
    P[0] = 0.0  # ensure self-prob is zero; caller will pass D[i,i]=0 at index 0 after sorting
    sumP = np.maximum(P.sum(), MACHINE_EPS)
    P = P / sumP
    H = np.log(sumP) + beta * np.sum(D_row * P)  # entropy in nats
    return H, P


def _binary_search_perplexity(D, perplexity, tol=1e-5, max_iter=50):
    """Compute conditional probabilities P_i|j per row via binary search on sigma.

    Parameters
    ----------
    D : ndarray, shape (n_samples, n_samples)
        Squared Euclidean distances.
    perplexity : float
        Target perplexity.
    Returns
    -------
    P_cond : ndarray, shape (n_samples, n_samples)
        Conditional probabilities (rows sum to 1, diag=0).
    """
    n = D.shape[0]
    logU = np.log(perplexity)

    # For numerical stability, we work on a copy with sorted neighbors per row
    P_cond = np.zeros((n, n), dtype=np.float64)
    betas = np.ones(n, dtype=np.float64)

    for i in range(n):
        # take distances to others (including self, which we'll zero later)
        Di = D[i].copy()
        Di[i] = 0.0

        beta_min, beta_max = -np.inf, np.inf
        beta = betas[i]

        # Evaluate entropy and P with current beta
        H, thisP = _hbeta(Di, beta)
        Hdiff = H - logU
        it = 0

        while np.abs(Hdiff) > tol and it < max_iter:
            if Hdiff > 0:
                # entropy too high -> distribution too broad -> increase beta
                beta_min = beta
                beta = beta * 2 if np.isinf(beta_max) else (beta + beta_max) / 2.0
            else:
                # entropy too low -> distribution too peaky -> decrease beta
                beta_max = beta
                beta = beta / 2 if np.isinf(beta_min) else (beta + beta_min) / 2.0

            H, thisP = _hbeta(Di, beta)
            Hdiff = H - logU
            it += 1

        betas[i] = beta
        thisP[i] = 0.0
        P_cond[i, :] = thisP

    # Symmetrize to joint P
    # Note: t-SNE uses P_ij = (P_i|j + P_j|i) / (2n)
    P = (P_cond + P_cond.T)
    P /= np.maximum(P.sum(), MACHINE_EPS)
    P = np.maximum(P, MACHINE_EPS)

    return P


def _joint_probabilities_exact(X, perplexity, metric="euclidean"):
    """Compute symmetric joint probabilities P_ij (dense, shape n x n)."""
    if metric == "euclidean":
        D = pairwise_distances(X, metric="euclidean", squared=True)
    else:
        D = pairwise_distances(X, metric=metric)
        # Square non-euclidean if you want to match legacy TSNE behavior:
        D **= 2

    if np.any(D < -1e-12):
        raise ValueError("Distances must be non-negative.")

    np.fill_diagonal(D, 0.0)
    P = _binary_search_perplexity(D, perplexity)
    # Small floor & renormalize
    P = np.maximum(P, MACHINE_EPS)
    P /= np.maximum(P.sum(), MACHINE_EPS)
    np.fill_diagonal(P, 0.0)
    return P


# ------------------------ Objective & Optimizer ------------------------ #

def _kl_divergence_exact(params, P, dof, n_samples, n_components):
    """KL divergence and gradient (exact, dense) using full P (n x n)."""
    Y = params.reshape(n_samples, n_components)

    # Student-t affinities in low-dim
    # Q_ij numerator = (1 + ||y_i - y_j||^2 / dof)^(-(dof+1)/2)
    # Commonly dof = 1 (Cauchy), so exponent = -1
    diff = Y[:, None, :] - Y[None, :, :]                      # (n, n, d)
    dist2 = np.sum(diff * diff, axis=2)                       # (n, n)
    inv = 1.0 / (1.0 + dist2 / dof)                           # (n, n)
    np.fill_diagonal(inv, 0.0)

    Q = inv / np.maximum(inv.sum(), MACHINE_EPS)
    Q = np.maximum(Q, MACHINE_EPS)

    # KL(P || Q)
    kl = np.sum(P * (np.log(P) - np.log(Q)))

    # Gradient: 4 * sum_j ( (P_ij - Q_ij) * inv_ij * (y_i - y_j) )
    # Note: dof factor cancels when using inv defined as above; for general dof:
    c = 2.0 * (dof + 1.0) / dof
    coeff = (P - Q) * inv                                     # (n, n)
    # For each i: grad_i = 4 * sum_j coeff_ij * (y_i - y_j)
    # We can compute with tensor operations:
    # term = sum_j coeff_ij * (y_i - y_j) = y_i * sum_j coeff_ij - sum_j coeff_ij * y_j
    sum_coeff = np.sum(coeff, axis=1)[:, None]                 # (n, 1)
    part1 = Y * sum_coeff                                      # (n, d)
    part2 = coeff @ Y                                          # (n, d)
    grad = 4.0 * (part1 - part2)
    grad *= c

    return kl, grad.ravel()


def _gradient_descent(objective, p0, n_iter, learning_rate,
                      momentum, n_iter_check=50, min_grad_norm=1e-7,
                      n_iter_without_progress=300, verbose=0,
                      callback_every=1):
    """Simple batch GD with momentum & gains; returns history at check points."""
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)

    best_error = np.inf
    best_iter = -1
    error = np.inf
    history = []  # list of (iter, error, embedding)

    tic = time()
    for it in range(n_iter):
        error, grad = objective(p)
        grad_norm = linalg.norm(grad)

        # adaptive gains
        inc = update * grad < 0.0
        dec = ~inc
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, 0.01, np.inf, out=gains)
        grad *= gains

        # momentum update
        update = momentum * update - learning_rate * grad
        p += update

        if (it + 1) % callback_every == 0:
            history.append((it + 1, float(error), p.copy()))

        if (it + 1) % n_iter_check == 0:
            if verbose >= 2:
                toc = time()
                print(f"[SN-tSNE] Iter {it+1}: KL={error:.6f} | ‖grad‖={grad_norm:.6f} "
                      f"({n_iter_check} iters in {toc - tic:.3f}s)")
                tic = toc

            if error < best_error:
                best_error = error
                best_iter = it
            elif it - best_iter > n_iter_without_progress:
                if verbose:
                    print(f"[SN-tSNE] Early stop (no progress in {n_iter_without_progress} iters).")
                break

            if grad_norm <= min_grad_norm:
                if verbose:
                    print(f"[SN-tSNE] Stop (grad norm {grad_norm:.3e}).")
                break

    return p, error, it, history


# ------------------------ Public API ------------------------ #

def trustworthiness(X, X_embedded, n_neighbors=5, metric="euclidean"):
    """Standard trustworthiness metric (0..1)."""
    D = pairwise_distances(X, metric=metric)
    np.fill_diagonal(D, np.inf)
    ind_X = np.argsort(D, axis=1)

    ind_Y = NearestNeighbors(n_neighbors=n_neighbors).fit(X_embedded).kneighbors(return_distance=False)
    n = X.shape[0]

    inv = np.zeros((n, n), dtype=int)
    order = np.arange(n + 1)
    inv[order[:-1, None], ind_X] = order[1:]
    ranks = inv[order[:-1, None], ind_Y] - n_neighbors
    t = np.sum(ranks[ranks > 0])
    return 1.0 - t * (2.0 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1)))


class SNTSNE:
    """Semi-non-negative t-SNE (SN-tSNE) using exact gradients and early exaggeration only.

    Parameters
    ----------
    n_components : int (default=2)
    perplexity : float (default=30.)
    early_exaggeration : float (default=12.)
    learning_rate : float or 'auto' (default='auto')
    n_iter : int (default=750)
    n_iter_without_progress : int (default=300)
    min_grad_norm : float (default=1e-7)
    init : {'random','pca'} or ndarray (default='random')
    metric : str (default='euclidean')
    random_state : int or None
    verbose : int
    """

    _EXPLORATION_N_ITER = 250
    _N_ITER_CHECK = 50

    def __init__(self,
                 n_components=2,
                 perplexity=30.0,
                 early_exaggeration=12.0,
                 learning_rate='auto',
                 n_iter=750,
                 n_iter_without_progress=300,
                 min_grad_norm=1e-7,
                 init='random',
                 metric='euclidean',
                 random_state=None,
                 verbose=0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.init = init
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

        # Outputs
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_iter_ = None
        self.embedding_history_ = None

    # ------------ main fit ------------- #
    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("Need at least 2 samples.")

        rng = check_random_state(self.random_state)

        # P (joint probabilities)
        if self.verbose:
            print("[SN-tSNE] Computing P (perplexity search)...")
        P = _joint_probabilities_exact(X, self.perplexity, metric=self.metric)

        # init Y
        if isinstance(self.init, np.ndarray):
            Y = np.asarray(self.init, dtype=np.float32)
            if Y.shape != (n_samples, self.n_components):
                raise ValueError("init array has wrong shape.")
        elif self.init == 'pca':
            if self.verbose:
                print("[SN-tSNE] PCA init...")
            Y = PCA(n_components=self.n_components, svd_solver="randomized",
                    random_state=rng).fit_transform(X).astype(np.float32)
            # semi-non-negative push (optional): square to reduce negatives
            Y = Y ** 2
        elif self.init == 'random':
            Y = (1e-4 * rng.randn(n_samples, self.n_components)).astype(np.float32)
            Y = Y ** 2
        else:
            raise ValueError("init must be 'pca', 'random', or an ndarray.")

        params = Y.ravel()
        dof = max(self.n_components - 1, 1)

        # learning rate
        if self.learning_rate == 'auto':
            lr = max(n_samples / self.early_exaggeration / 4, 50.0)
        else:
            lr = float(self.learning_rate)

        # Phase 1: early exaggeration
        if self.verbose:
            print("[SN-tSNE] Early exaggeration phase...")
        P_exag = P * self.early_exaggeration

        def obj_EE(p):
            return _kl_divergence_exact(p, P_exag, dof, n_samples, self.n_components)

        p_opt, err, it, hist = _gradient_descent(
            objective=obj_EE,
            p0=params,
            n_iter=min(self._EXPLORATION_N_ITER, self.n_iter),
            learning_rate=lr,
            momentum=0.5,
            n_iter_check=self._N_ITER_CHECK,
            min_grad_norm=self.min_grad_norm,
            n_iter_without_progress=self.n_iter_without_progress,
            verbose=self.verbose,
            callback_every=10
        )

        # Phase 2: final
        remaining = max(self.n_iter - (it + 1), 0)
        if remaining > 0:
            if self.verbose:
                print("[SN-tSNE] Final optimization phase...")

            def obj_final(p):
                return _kl_divergence_exact(p, P, dof, n_samples, self.n_components)

            p_opt, err, it2, hist2 = _gradient_descent(
                objective=obj_final,
                p0=p_opt,
                n_iter=remaining,
                learning_rate=lr,
                momentum=0.8,
                n_iter_check=self._N_ITER_CHECK,
                min_grad_norm=self.min_grad_norm,
                n_iter_without_progress=self.n_iter_without_progress,
                verbose=self.verbose,
                callback_every=10
            )
            hist.extend([(self._EXPLORATION_N_ITER + h[0], h[1], h[2]) for h in hist2])
            it = self._EXPLORATION_N_ITER + it2 + 1  # total iterations counted

        Y_final = p_opt.reshape(n_samples, self.n_components)

        self.embedding_ = Y_final
        self.kl_divergence_ = float(err)
        self.n_iter_ = int(it)
        # store only embeddings (drop KL) for history
        self.embedding_history_ = [emb.reshape(n_samples, self.n_components).copy()
                                   for (_, _, emb) in hist]

        return self.embedding_, self.kl_divergence_, self.embedding_history_

    # convenience alias
    def fit_transform(self, X):
        emb, _, _ = self.fit(X)
        return emb
