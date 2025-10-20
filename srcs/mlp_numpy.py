
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

def he_uniform(shape, rng):
    limit = math.sqrt(6 / shape[0])
    return rng.uniform(-limit, limit, size=shape)

def xavier_uniform(shape, rng):
    limit = math.sqrt(6 / (shape[0] + shape[1]))
    return rng.uniform(-limit, limit, size=shape)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    # derivative wrt output of sigmoid
    return y * (1 - y)

def relu(x):
    return np.maximum(0, x)

def drelu(y):
    return (y > 0).astype(y.dtype)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

@dataclass
class Layer:
    W: np.ndarray
    b: np.ndarray
    activation: str

    def forward(self, x):
        z = x @ self.W + self.b
        if self.activation == "sigmoid":
            return sigmoid(z), z
        elif self.activation == "relu":
            return relu(z), z
        elif self.activation == "linear":
            return z, z
        elif self.activation == "softmax":
            return softmax(z), z
        else:
            raise ValueError(f"Unknown activation {self.activation}")

    def activation_deriv(self, y):
        if self.activation == "sigmoid":
            return dsigmoid(y)
        elif self.activation == "relu":
            return drelu(y)
        elif self.activation in ("linear", "softmax"):
            # handled in loss gradient for softmax+crossentropy; linear derivative is 1
            return np.ones_like(y)
        else:
            raise ValueError(f"Unknown activation {self.activation}")

class MLP:
    def __init__(self, layer_sizes: List[int], activations: List[str], seed: int = 42):
        assert len(layer_sizes) >= 2
        assert len(activations) == len(layer_sizes) - 1
        rng = np.random.default_rng(seed)
        self.layers: List[Layer] = []
        for in_size, out_size, act in zip(layer_sizes[:-1], layer_sizes[1:], activations):
            W = he_uniform((in_size, out_size), rng) if act in ("relu",) else xavier_uniform((in_size, out_size), rng)
            b = np.zeros((1, out_size))
            self.layers.append(Layer(W=W, b=b, activation=act))

    def forward(self, x) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        a = x
        activations = [a]
        zs = []
        for layer in self.layers:
            a, z = layer.forward(a)
            activations.append(a)
            zs.append(z)
        return activations, zs

    def predict_proba(self, x):
        a, _ = self.forward(x)
        return a[-1]

    def save(self, path, meta: Dict):
        np.savez(path, **{f"W{i}": l.W for i, l in enumerate(self.layers)},
                       **{f"b{i}": l.b for i, l in enumerate(self.layers)},
                       meta=np.array([json.dumps(meta)]))

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(data["meta"][0].item() if isinstance(data["meta"][0], dict) else data["meta"][0])
        layer_sizes = meta["layer_sizes"]
        activations = meta["activations"]
        mlp = MLP(layer_sizes, activations, seed=meta.get("seed", 42))
        for i, layer in enumerate(mlp.layers):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]
        return mlp, meta

def one_hot(y, num_classes):
    Y = np.zeros((len(y), num_classes))
    Y[np.arange(len(y)), y] = 1
    return Y

def cross_entropy(pred, Y):
    eps = 1e-12
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(np.sum(Y * np.log(pred), axis=1))

def accuracy(pred, Y):
    return np.mean(np.argmax(pred, axis=1) == np.argmax(Y, axis=1))

def gd_update(params, grads, lr):
    for p, g in zip(params, grads):
        p -= lr * g

def train(mlp: MLP, X_train, Y_train, X_val, Y_val, epochs=50, lr=0.03, batch_size=32, verbose=True):
    n = len(X_train)
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    rng = np.random.default_rng(0)
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n)
        X_train = X_train[idx]; Y_train = Y_train[idx]
        # mini-batch
        for start in range(0, n, batch_size):
            end = start + batch_size
            xb = X_train[start:end]
            yb = Y_train[start:end]

            # forward
            acts, zs = mlp.forward(xb)
            # backward (softmax + cross-entropy gradient)
            delta = (acts[-1] - yb) / len(xb)

            dWs = []
            dbs = []
            for i in reversed(range(len(mlp.layers))):
                a_prev = acts[i]
                dW = a_prev.T @ delta
                db = np.sum(delta, axis=0, keepdims=True)
                dWs.insert(0, dW)
                dbs.insert(0, db)

                if i != 0:
                    # propagate to previous layer
                    prev = mlp.layers[i-1]
                    delta = (delta @ mlp.layers[i].W.T) * mlp.layers[i-1].activation_deriv(acts[i])

            # update
            for layer, dW, db in zip(mlp.layers, dWs, dbs):
                layer.W -= lr * dW
                layer.b -= lr * db

        # epoch metrics
        train_pred = mlp.predict_proba(X_train)
        val_pred = mlp.predict_proba(X_val)
        tr_loss = cross_entropy(train_pred, Y_train)
        va_loss = cross_entropy(val_pred, Y_val)
        tr_acc = accuracy(train_pred, Y_train)
        va_acc = accuracy(val_pred, Y_val)
        history["loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        if verbose:
            print(f"epoch {epoch:02d} - loss: {tr_loss:.4f} - val_loss: {va_loss:.4f} - acc: {tr_acc:.4f} - val_acc: {va_acc:.4f}")
    return history
