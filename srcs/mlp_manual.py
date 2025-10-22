from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import os


#init
def he_uniform(shape, rng):
    fan_in = shape[0]
    limit = math.sqrt(6.0 / max(1, fan_in))
    return rng.uniform(-limit, limit, size=shape)


def xavier_uniform(shape, rng):
    fan_in, fan_out = shape[0], shape[1]
    limit = math.sqrt(6.0 / max(1, fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


def relu(x): 
    return np.maximum(0.0, x)


def relu_grad_from_z(z):
    # z<0: 0, z>0: 1
    return (z > 0.0).astype(z.dtype)


def softmax_stable(z):
    z = z - z.max(axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)


def cross_entropy(pred, Y, eps: float = 1e-12):
    pred = np.clip(pred, eps, 1.0)
    return -np.mean(np.sum(Y * np.log(pred), axis=1))


def accuracy(pred, Y):
    return (pred.argmax(axis=1) == Y.argmax(axis=1)).mean()


# layers
@dataclass
class Layer:
    W: np.ndarray
    b: np.ndarray
    activation: str  # relu or softmax

    def forward(self, a_prev):
        z = a_prev @ self.W + self.b
        if self.activation == "relu":
            a = relu(z)
        elif self.activation == "softmax":
            a = softmax_stable(z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return a, z


class MLP:
    def __init__(self, layer_sizes: List[int], activations: List[str], seed: int = 42):
        """
        layer_sizes: [D_in, H1, ..., D_out]
        activations: ["relu", ..., "softmax"]  (softmax for the last)
        """
        assert len(layer_sizes) >= 2
        assert len(activations) == len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)

        self.layers: List[Layer] = []
        for in_size, out_size, act in zip(layer_sizes[:-1], layer_sizes[1:], activations):
            if act == "relu":
                W = he_uniform((in_size, out_size), rng)
            else:
                W = xavier_uniform((in_size, out_size), rng)
            b = np.zeros((1, out_size), dtype=W.dtype)
            self.layers.append(Layer(W=W.astype(np.float64), b=b.astype(np.float64), activation=act))

    # forward for backward
    def forward(self, X) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        a = X.astype(np.float64)
        activs = [a]       # A0 = input
        zs = []            # z of each layer
        for layer in self.layers:
            a, z = layer.forward(a)
            activs.append(a)
            zs.append(z)
        return activs, zs

    def predict_proba(self, X):
        a, _ = self.forward(X)
        return a[-1]


# gradient
def backward(mlp: MLP, activs: List[np.ndarray], zs: List[np.ndarray], Y_batch: np.ndarray):
    """
    Softmax + CE: output layer delta = (pred - Y)
    hidden layer: delta = (delta_next @ W_next.T) * ReLU'(z_prev)
    return: dWs, dbs
    """
    L = len(mlp.layers)
    B = Y_batch.shape[0]

    # output layer
    pred = activs[-1]              # A_L
    delta = (pred - Y_batch) / B   # mean BATCH

    dWs = [None] * L
    dbs = [None] * L

    for i in reversed(range(L)):
        a_prev = activs[i]         # A_{i}
        dW = a_prev.T @ delta
        db = delta.sum(axis=0, keepdims=True)
        dWs[i] = dW
        dbs[i] = db

        if i != 0:
            W = mlp.layers[i].W
            z_prev = zs[i-1]
            delta = (delta @ W.T) * relu_grad_from_z(z_prev)

    return dWs, dbs


def clip_gradients_(dWs, dbs, max_norm: Optional[float]):
    if not max_norm or max_norm <= 0:
        return

    total_sq = 0.0

    for dW, db in zip(dWs, dbs):
        total_sq += np.sum(dW * dW) + np.sum(db * db)
    total_norm = math.sqrt(total_sq) + 1e-12

    if total_norm > max_norm:
        scale = max_norm / total_norm

        for i in range(len(dWs)):
            dWs[i] *= scale
            dbs[i] *= scale


def sgd_step_(mlp: MLP, dWs, dbs, lr: float, weight_decay: float = 0.0):
    # SGD + L2(weight decay)
    for layer, dW, db in zip(mlp.layers, dWs, dbs):
        if weight_decay and weight_decay > 0.0:
            layer.W -= lr * (dW + weight_decay * layer.W)
        else:
            layer.W -= lr * dW
        layer.b -= lr * db


# training main loop
def train_loop(
    mlp: MLP,
    X_tr: np.ndarray, Y_tr: np.ndarray,
    X_va: np.ndarray, Y_va: np.ndarray,
    epochs: int = 30, lr: float = 0.05, batch_size: int = 64,
    seed: int = 42, patience: int = 5,
    lr_decay_at: int = 10, lr_decay_factor: float = 0.5,
    weight_decay: float = 0.0, max_grad_norm: float = 0.0,
    verbose: bool = True
):
    """
    early stopping: based on val_loss,
                    stop and recover the best check point
                    when runout of patience
    """
    rng = np.random.default_rng(int(seed))
    N = X_tr.shape[0]
    hist = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    best_val = float("inf")
    best_snapshot = None
    best_epoch = -1
    left = patience

    lr_curr = float(lr)

    for epoch in range(1, epochs + 1):
        # shuffle EPOCH
        idx = rng.permutation(N)
        X_tr = X_tr[idx]; Y_tr = Y_tr[idx]

        # mini BATCH training
        for s in range(0, N, batch_size):
            e = s + batch_size
            xb = X_tr[s:e]; yb = Y_tr[s:e]

            activs, zs = mlp.forward(xb)
            dWs, dbs = backward(mlp, activs, zs, yb)

            # selective gradient cliping
            clip_gradients_(dWs, dbs, max_norm=max_grad_norm)

            # update with L2
            sgd_step_(mlp, dWs, dbs, lr=lr_curr, weight_decay=weight_decay)

        # eval EPOCH
        p_tr = mlp.predict_proba(X_tr)
        p_va = mlp.predict_proba(X_va)

        loss_tr = cross_entropy(p_tr, Y_tr)
        loss_va = cross_entropy(p_va, Y_va)
        acc_tr = accuracy(p_tr, Y_tr)
        acc_va = accuracy(p_va, Y_va)

        hist["loss"].append(loss_tr)
        hist["val_loss"].append(loss_va)
        hist["acc"].append(acc_tr)
        hist["val_acc"].append(acc_va)

        if verbose:
            print(f"[{epoch:02d}] loss={loss_tr:.4f} acc={acc_tr:.4f} | val_loss={loss_va:.4f} val_acc={acc_va:.4f}")

        # early fininshing
        if loss_va < best_val - 1e-8:
            best_val = loss_va
            best_snapshot = [(layer.W.copy(), layer.b.copy()) for layer in mlp.layers]
            best_epoch = epoch
            left = patience
        else:
            left -= 1
            if left == 0:
                if verbose:
                    print(f"[early-stopping] epoch={epoch}, best@{best_epoch} val_loss={best_val:.4f}")
                # the best W b
                for (W, b), layer in zip(best_snapshot, mlp.layers):
                    layer.W, layer.b = W, b
                break

        # LR decreasing
        if lr_decay_at and epoch == lr_decay_at:
            lr_curr *= float(lr_decay_factor)
            if verbose:
                print(f"[lr-decay] lr -> {lr_curr}")

    return hist, (best_epoch if best_epoch != -1 else epoch)


# save / load 
def save_npz(path: str, mlp: MLP, meta: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blobs = {}
    for i, layer in enumerate(mlp.layers, start=1):
        blobs[f"W{i}"] = layer.W
        blobs[f"b{i}"] = layer.b

    # meta > python dict type
    blobs["meta"] = np.array([meta], dtype=object)
    np.savez(path, **blobs)


def load_npz(path: str) -> Tuple[MLP, Dict]:
    npz = np.load(path, allow_pickle=True)
    meta = npz["meta"].item() if "meta" in npz.files else {}
    layer_sizes = meta["layer_sizes"]
    activations = meta["activations"]
    seed = int(meta.get("seed", 42))

    mlp = MLP(layer_sizes, activations, seed=seed)
    L = len(layer_sizes) - 1
    for i in range(1, L + 1):
        mlp.layers[i-1].W = npz[f"W{i}"]
        mlp.layers[i-1].b = npz[f"b{i}"]
    return mlp, meta
