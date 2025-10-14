# SN_tSNE.py
# Semi-Non-Negative t-SNE (SN-tSNE) that mirrors PN_tSNE.py’s behavior:
# - SAME objective & gradient (with per-point adaptive scaling at the end)
# - SAME optimizer shape (gains, momentum, clipping)
# - Returns (embedding, error, history) to match Test_SN_tSNE.py
# - Semi-non-negative: project embedding to >= 0 every iteration
#
# Tip: start with learning_rate=1.0, perplexity=30, n_iter=1000 on 'digits'

import warnings
from time import time
import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

MACHINE_EPS = 1e-12

# ------------------------------ P computation ------------------------------ #

def _hbeta(D_row, beta):
    P = np.exp(-D_row * beta)
    P[0] = 0.0  # self prob
    sumP = np.maximum(P.sum(), MACHINE_EPS)
    P /= sumP
    H = np.log(sumP) + beta * np.sum(D_row * P)
    return H, P

def _binary_search_perplexity(D, perplexity, tol=1e-5, max_iter=50, verbose=0):
    n = D.shape[0]
    logU = np.log(perplexity)
    P_cond = np.zeros((n, n), dtype=np.float64)
    betas = np.ones(n, dtype=np.float64)

    for i in range(n):
        Di = D[i].copy()
        Di[i] = 0.0

        beta_min, beta_max = -np.inf, np.inf
        beta = betas[i]
        H, thisP = _hbeta(Di, beta)
        Hdiff = H - logU
        it = 0

        while np.abs(Hdiff) > tol and it < max_iter:
            if Hdiff > 0:
                beta_min = beta
                beta = beta * 2 if np.isinf(beta_max) else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2 if np.isinf(beta_min) else (beta + beta_min) / 2.0
            H, thisP = _hbeta(Di, beta)
            Hdiff = H - logU
            it += 1

        betas[i] = beta
        thisP[i] = 0.0
        P_cond[i, :] = thisP

    P = P_cond + P_cond.T
    sumP = np.maximum(P.sum(), MACHINE_EPS)
    P = np.maximum(squareform(P) / sumP, MACHINE_EPS)
    return P

def _joint_probabilities(X, perplexity, metric="euclidean", verbose=0):
    if metric == "euclidean":
        D = pairwise_distances(X, metric="euclidean", squared=True)
    else:
        D = pairwise_distances(X, metric=metric)
        D **= 2
    np.fill_diagonal(D, 0.0)
    return _binary_search_perplexity(D, perplexity, verbose=verbose)

# ----------------------- KL divergence + PN-style grad ---------------------- #

def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components,
                   skip_num_points=0, compute_error=True):
    """
    Identical structure to PN_tSNE.py:
      - Student-t Q from pairwise distances
      - KL objective
      - Gradient = sum_j ((P - Q) * t-kernel term) * (y_i - y_j)
      - Per-point LR at the tail: lr_i = 1 / (sum_j 4*PQd_ij + eps)
    """
    eps = 1e-12
    max_loss = 1e6

    X_embedded = params.reshape(n_samples, n_components)

    # Student-t kernel (dof typically = 1 for t-SNE)
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0

    # Q
    Q = np.maximum(dist / (2.0 * np.sum(dist)), eps)

    # KL
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, eps) / Q))
    kl_divergence = np.clip(kl_divergence, 0, max_loss)

    # Gradient
    grad = np.zeros((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(PQd[i], X_embedded[i] - X_embedded)

    lr = 1.0 / (np.sum(4.0 * PQd, axis=1) + eps)   # per-point scaling (your tail)
    grad = (grad.T * lr).T
    grad *= 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom

    return kl_divergence, grad.ravel()

# ------------------------------- Optimizer --------------------------------- #

def _gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=50,
                      n_iter_without_progress=300,
                      momentum=0.8,
                      learning_rate=1.0,     # PN default they used
                      min_gain=0.01,
                      min_grad_norm=1e-7,
                      semi_nonneg=True,      # <— SN: keep non-negative
                      verbose=1,
                      args=None, kwargs=None):
    """
    Matches PN_tSNE style: gains, momentum, clipping, periodic checks.
    Adds semi-non-negative projection after each update.
    """
    max_update = 1e2
    if args is None: args = []
    if kwargs is None: kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it
    embedding_history = []

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0

        kwargs["compute_error"] = check_convergence or i == n_iter - 1
        error, grad = objective(p, *args, **kwargs)

        grad_norm = linalg.norm(grad)
        grad = np.clip(grad, -max_update, max_update)

        inc = update * grad < 0.0
        dec = ~inc
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains

        update = momentum * update - grad * learning_rate
        update = np.clip(update, -max_update, max_update)
        p += update

        # ---- SN projection: keep embedding non-negative ----
        if semi_nonneg:
            X_emb = p.reshape(-1, int(len(p) / (len(p0) // (p0.size // (len(p0)//2) if len(p0) else 2)) ) )
            # Simpler: reshape with known n_components
        # Safer reshape using args (we know n_samples, n_components there)
        n_samples, n_components = args[2], args[3]
        X_emb = p.reshape(n_samples, n_components)
        X_emb = np.maximum(X_emb, 1e-12)
        p = X_emb.ravel()
        # ----------------------------------------------------

        # save snapshot for history (optional, every check)
        if check_convergence:
            embedding_history.append(X_emb.copy())

            toc = time()
            if verbose >= 1:
                print(f"[SN-tSNE] iter {i+1:4d} | KL={error:.6f}")
            tic = toc

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 1:
                    print(f"[SN-tSNE] Early stop: no progress in {n_iter_without_progress} iters.")
                break

            if grad_norm <= min_grad_norm:
                if verbose >= 1:
                    print(f"[SN-tSNE] Stop (grad norm {grad_norm:.3e}).")
                break

    return p, error, i, embedding_history

# --------------------------------- Model ----------------------------------- #

class SNTSNE(BaseEstimator):
    _EXPLORATION_N_ITER = 250
    _N_ITER_CHECK = 50

    def __init__(self,
                 n_components=2,
                 perplexity=30.0,
                 early_exaggeration=12.0,
                 learning_rate=1.0,           # match PN_tSNE.py default they used
                 n_iter=1000,
                 n_iter_without_progress=300,
                 min_grad_norm=1e-7,
                 metric="euclidean",
                 init="pca",                   # PN’s preferred start (we square below)
                 verbose=1,
                 random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state

    def fit_transform(self, X):
        E, _, _ = self.fit(X)
        return E

    def fit(self, X, skip_num_points=0):
        # distances
        if self.verbose:
            print("[SN-tSNE] Computing pairwise distances...")
        if self.metric == "euclidean":
            distances = pairwise_distances(X, metric=self.metric, squared=True)
        else:
            distances = pairwise_distances(X, metric=self.metric)
        if np.any(distances < 0):
            raise ValueError("All distances must be non-negative.")

        # P
        P = _joint_probabilities(X, self.perplexity, metric=self.metric, verbose=self.verbose)
        assert np.all(np.isfinite(P)), "All probabilities should be finite"
        assert np.all(P >= 0), "All probabilities should be non-negative"
        assert np.all(P <= 1), "All probabilities should be <= 1"

        # init
        n_samples = X.shape[0]
        if self.init == "pca":
            X_embedded = PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(X).astype(np.float32, copy=False)
            # PN trick: square to land in non-negative region
            X_embedded = (X_embedded ** 2).astype(np.float32, copy=False)
        elif self.init == "random":
            rng = np.random.RandomState(self.random_state)
            X_embedded = 1e-4 * rng.randn(n_samples, self.n_components).astype(np.float32)
            X_embedded = (X_embedded ** 2).astype(np.float32, copy=False)
        else:
            raise ValueError("init must be 'pca' or 'random'")

        degrees_of_freedom = max(self.n_components - 1, 1)
        params = X_embedded.ravel()

        # --- Phase 1: Early exaggeration only (match PN behavior closely) ---
        P_exag = P * self.early_exaggeration
        obj = _kl_divergence
        opt = dict(
            it=0,
            n_iter=self._EXPLORATION_N_ITER if self.n_iter >= self._EXPLORATION_N_ITER else self.n_iter,
            n_iter_check=self._N_ITER_CHECK,
            n_iter_without_progress=self._EXPLORATION_N_ITER,  # like PN’s exploration gating
            momentum=0.5,
            learning_rate=self.learning_rate,
            min_grad_norm=self.min_grad_norm,
            verbose=self.verbose,
            args=[P_exag, degrees_of_freedom, n_samples, self.n_components],
            kwargs=dict(skip_num_points=skip_num_points),
            semi_nonneg=True,
        )
        params, error, it, embedding_history = _gradient_descent(obj, params, **opt)

        # If more iters requested, continue without exaggeration (optional)
        remaining = self.n_iter - opt["n_iter"]
        if remaining > 0:
            P_norm = P  # remove exaggeration
            opt2 = dict(
                it=it + 1,
                n_iter=it + 1 + remaining,
                n_iter_check=self._N_ITER_CHECK,
                n_iter_without_progress=self.n_iter_without_progress,
                momentum=0.8,
                learning_rate=self.learning_rate,
                min_grad_norm=self.min_grad_norm,
                verbose=self.verbose,
                args=[P_norm, degrees_of_freedom, n_samples, self.n_components],
                kwargs=dict(skip_num_points=skip_num_points),
                semi_nonneg=True,
            )
            params, error, it, embedding_history2 = _gradient_descent(obj, params, **opt2)
            embedding_history.extend(embedding_history2)

        X_embedded = params.reshape(n_samples, self.n_components)
        X_embedded = np.maximum(X_embedded, 1e-12)  # final SN clamp

        self.X_embedded = X_embedded
        self.embedding_history = embedding_history
        return self.X_embedded, error, self.embedding_history
