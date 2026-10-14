"""Transparent supervised baseline; intentionally not described as deep learning."""
import numpy as np


class RidgeSurrogate:
    """Dual ridge regression, early design variables -> final design variables.

    Mean and scaling are fitted ONLY on training cases. Hyperparameters must be
    selected using validation cases. No augmentation in this first benchmark.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, inputs, targets):
        x, y = np.asarray(inputs), np.asarray(targets)
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 0.05)
        self.train = (x-self.mean)/self.scale / np.sqrt(x.shape[1])
        self.target_mean = y.mean(axis=0)
        self.weights = np.linalg.solve(self.train@self.train.T+self.alpha*np.eye(len(x)), y-self.target_mean)
        return self

    def predict(self, inputs):
        x = (np.atleast_2d(inputs)-self.mean)/self.scale / np.sqrt(self.mean.size)
        return x@self.train.T@self.weights+self.target_mean


def split_cases(n, seed=3017, groups=None):
    """Case-group split; mirrored/equivalent cases can share a group identifier."""
    if n < 10:
        raise ValueError("At least ten distinct cases are required")
    groups = np.arange(n) if groups is None else np.asarray(groups)
    if len(groups) != n or len(np.unique(groups)) < 5:
        raise ValueError("Need >= 5 groups and one group label per case")
    ids = np.random.default_rng(seed).permutation(np.unique(groups))
    a, b = int(0.6*len(ids)), int(0.8*len(ids))
    return {key: np.flatnonzero(np.isin(groups, subset))
            for key,subset in zip(('train','validation','test'),(ids[:a],ids[a:b],ids[b:]))}
