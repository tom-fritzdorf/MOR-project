"""Hand-rolled NumPy feed-forward network for the parametric PODNN (Task 2).

Implements PODNN offline step 4 from the class notes (Hesthaven & Ubbiali
2018, `3.pdf`): train a feed-forward network ``N_theta: mu -> c(mu)``
minimizing MSE against the POD coefficients Task 1 already projected from
the FOM snapshots. No external ML dependency is used -- ``mor_env`` has no
scikit-learn/PyTorch/TensorFlow and no pip, and a plain MLP trained with
Adam on ~130 samples is exactly the scale the class notes' own PODNN example
uses ("two hidden layers... seconds on a laptop"), so a from-scratch
implementation is both sufficient and consistent with the rest of this
dependency-light codebase (numpy/scipy/pandas/matplotlib/seaborn only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ==========================================================================
# Standardization
# ==========================================================================
@dataclass
class StandardScaler:
    """Per-feature zero-mean/unit-std standardization."""
    mean_: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale_: np.ndarray = field(default_factory=lambda: np.ones(0))

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.mean_


# ==========================================================================
# Feed-forward network
# ==========================================================================
class FeedForwardNet:
    """Plain MLP: tanh hidden layers, linear output, Adam optimizer.

    Manual forward/backward (no autodiff) -- tractable because the network
    is small (a few hundred parameters) and trained full-batch on ~130
    samples, matching the PODNN literature's own network scale.
    """

    def __init__(self, layer_sizes: List[int], seed: int = 42) -> None:
        self.layer_sizes = list(layer_sizes)
        rng = np.random.default_rng(seed)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))  # Glorot uniform (tanh-friendly)
            self.W.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(bb) for bb in self.b]
        self._vb = [np.zeros_like(bb) for bb in self.b]
        self._t = 0

    # ---- forward ----
    def forward(self, X: np.ndarray
               ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Returns ``(output, activations incl. input, pre-activations)``."""
        A = X
        activations = [A]
        preacts: List[np.ndarray] = []
        n_layers = len(self.W)
        for i in range(n_layers):
            Z = A @ self.W[i] + self.b[i]
            preacts.append(Z)
            A = np.tanh(Z) if i < n_layers - 1 else Z  # linear output layer
            activations.append(A)
        return A, activations, preacts

    def predict(self, X: np.ndarray) -> np.ndarray:
        out, _, _ = self.forward(X)
        return out

    # ---- backward ----
    def _backward(self, activations: List[np.ndarray], preacts: List[np.ndarray],
                  dOut: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        n_layers = len(self.W)
        dW: List[Optional[np.ndarray]] = [None] * n_layers
        db: List[Optional[np.ndarray]] = [None] * n_layers
        dA = dOut
        for i in reversed(range(n_layers)):
            dZ = dA if i == n_layers - 1 else dA * (1.0 - np.tanh(preacts[i]) ** 2)
            dW[i] = activations[i].T @ dZ
            db[i] = dZ.sum(axis=0)
            dA = dZ @ self.W[i].T
        return dW, db  # type: ignore[return-value]

    # ---- Adam step ----
    def _adam_step(self, dW: List[np.ndarray], db: List[np.ndarray],
                   lr: float, beta1: float = 0.9, beta2: float = 0.999,
                   eps: float = 1e-8) -> None:
        self._t += 1
        for i in range(len(self.W)):
            self._mW[i] = beta1 * self._mW[i] + (1 - beta1) * dW[i]
            self._vW[i] = beta2 * self._vW[i] + (1 - beta2) * dW[i] ** 2
            mW_hat = self._mW[i] / (1 - beta1 ** self._t)
            vW_hat = self._vW[i] / (1 - beta2 ** self._t)
            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

            self._mb[i] = beta1 * self._mb[i] + (1 - beta1) * db[i]
            self._vb[i] = beta2 * self._vb[i] + (1 - beta2) * db[i] ** 2
            mb_hat = self._mb[i] / (1 - beta1 ** self._t)
            vb_hat = self._vb[i] / (1 - beta2 ** self._t)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    # ---- one full-batch training step ----
    def train_step(self, X: np.ndarray, Y: np.ndarray, lr: float,
                   weight_decay: float = 0.0) -> float:
        pred, activations, preacts = self.forward(X)
        diff = pred - Y
        loss = float(np.mean(diff ** 2))
        dOut = (2.0 / diff.size) * diff
        dW, db = self._backward(activations, preacts, dOut)
        if weight_decay > 0.0:
            dW = [dw + weight_decay * w for dw, w in zip(dW, self.W)]
        self._adam_step(dW, db, lr)
        return loss

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        pred = self.predict(X)
        return float(np.mean((pred - Y) ** 2))

    # ---- weight I/O ----
    def get_weights(self) -> List[np.ndarray]:
        return [w.copy() for w in self.W] + [bb.copy() for bb in self.b]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        n = len(self.W)
        self.W = [w.copy() for w in weights[:n]]
        self.b = [bb.copy() for bb in weights[n:]]


def train_val_split(params: np.ndarray, coeffs: np.ndarray,
                    val_frac: float = 0.2, seed: int = 42
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``params`` (M,P), ``coeffs`` (M,R) -> (mu_tr, c_tr, mu_val, c_val)."""
    M = params.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(M)
    n_val = max(1, int(round(val_frac * M)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (params[train_idx], coeffs[train_idx],
            params[val_idx], coeffs[val_idx])


# ==========================================================================
# PODNN model: net + input/output standardization + early stopping
# ==========================================================================
class PODNNModel:
    """mu -> POD coefficients regressor (Hesthaven & Ubbiali 2018 PODNN).

    Wraps :class:`FeedForwardNet` with input/output standardization (output
    standardization matters because POD coefficient magnitudes decay sharply
    across modes -- the leading singular values would otherwise dominate a
    raw MSE) and early stopping on a held-out validation split.
    """

    def __init__(self, input_dim: int, output_dim: int,
                hidden_sizes: Optional[List[int]] = None,
                seed: int = 42) -> None:
        hidden_sizes = hidden_sizes if hidden_sizes is not None else [128, 128, 64]
        self.layer_sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        self.net = FeedForwardNet(self.layer_sizes, seed=seed)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self, mu_train: np.ndarray, c_train: np.ndarray,
           mu_val: np.ndarray, c_val: np.ndarray,
           epochs: int = 5000, lr: float = 1e-3,
           patience: int = 300, weight_decay: float = 0.0,
           verbose: bool = False
           ) -> Tuple[np.ndarray, np.ndarray]:
        self.x_scaler.fit(mu_train)
        self.y_scaler.fit(c_train)
        Xtr = self.x_scaler.transform(mu_train)
        Ytr = self.y_scaler.transform(c_train)
        Xval = self.x_scaler.transform(mu_val)
        Yval = self.y_scaler.transform(c_val)

        train_hist = np.zeros(epochs)
        val_hist = np.zeros(epochs)
        best_val = np.inf
        best_weights = self.net.get_weights()
        best_epoch = 0
        last_epoch = epochs - 1
        for ep in range(epochs):
            tr_loss = self.net.train_step(Xtr, Ytr, lr, weight_decay=weight_decay)
            val_loss = self.net.loss(Xval, Yval)
            train_hist[ep] = tr_loss
            val_hist[ep] = val_loss
            if val_loss < best_val - 1e-12:
                best_val = val_loss
                best_weights = self.net.get_weights()
                best_epoch = ep
            if verbose and ep % 500 == 0:
                print(f"  epoch {ep:5d}  train {tr_loss:.3e}  val {val_loss:.3e}")
            if ep - best_epoch > patience:
                last_epoch = ep
                break
        self.net.set_weights(best_weights)
        return train_hist[: last_epoch + 1], val_hist[: last_epoch + 1]

    def predict(self, mu: np.ndarray) -> np.ndarray:
        Xs = self.x_scaler.transform(mu)
        Ys = self.net.predict(Xs)
        return self.y_scaler.inverse_transform(Ys)

    # ---- I/O ----
    def save(self, path: Path) -> None:
        weights = self.net.get_weights()
        n = len(self.net.W)
        np.savez_compressed(
            path,
            layer_sizes=np.array(self.layer_sizes),
            n_weight_layers=np.array(n),
            **{f"W{i}": weights[i] for i in range(n)},
            **{f"b{i}": weights[n + i] for i in range(n)},
            x_mean=self.x_scaler.mean_, x_scale=self.x_scaler.scale_,
            y_mean=self.y_scaler.mean_, y_scale=self.y_scaler.scale_,
        )

    @classmethod
    def load(cls, path: Path) -> "PODNNModel":
        d = np.load(path, allow_pickle=False)
        layer_sizes = [int(v) for v in d["layer_sizes"]]
        n = int(d["n_weight_layers"])
        model = cls(layer_sizes[0], layer_sizes[-1], hidden_sizes=layer_sizes[1:-1])
        weights = [d[f"W{i}"] for i in range(n)] + [d[f"b{i}"] for i in range(n)]
        model.net.set_weights(weights)
        model.x_scaler.mean_ = d["x_mean"]; model.x_scaler.scale_ = d["x_scale"]
        model.y_scaler.mean_ = d["y_mean"]; model.y_scaler.scale_ = d["y_scale"]
        return model
