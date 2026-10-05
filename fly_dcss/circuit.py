"""Outgoing CSR adapter and bounded rate dynamics with local eligibility.

Loading convention adapted from jppaquet/flybrain (MIT); see THIRD_PARTY.md.
All rates, weights and time units below are dimensionless modeling choices.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def _vector(value, name, dtype=None):
    value = np.asarray(value)
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if dtype is int and value.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    return value.astype(np.int64 if dtype is int else np.float64, copy=True)


@dataclass(frozen=True)
class Graph:
    """Validated outgoing CSR: row=pre, indices=post, weight=magnitude.

    Arrays are copied and read-only. ID namespace is supplied by the loader;
    converted NPZ IDs are upstream body IDs, fly-hero IDs are local indices.
    """
    indptr: np.ndarray
    indices: np.ndarray
    weight: np.ndarray
    sign: np.ndarray
    body: np.ndarray

    def __post_init__(self):
        for name in ("indptr", "indices", "weight", "sign", "body"):
            a = _vector(getattr(self, name), name,
                        int if name in ("indptr", "indices", "body") else None)
            object.__setattr__(self, name, a)
        n = len(self.body)
        if n == 0 or len(self.indptr) != n + 1 or len(self.sign) != n:
            raise ValueError("inconsistent or empty neuron arrays")
        if (self.indptr[0] != 0 or np.any(np.diff(self.indptr) < 0)
                or self.indptr[-1] != len(self.indices)
                or len(self.weight) != len(self.indices)):
            raise ValueError("invalid CSR pointers or edge lengths")
        if np.any((self.indices < 0) | (self.indices >= n)):
            raise ValueError("post index out of range")
        if not np.all(np.isfinite(self.weight)) or np.any(self.weight < 0):
            raise ValueError("weights must be finite nonnegative magnitudes")
        if not np.all(np.isin(self.sign, [-1, 1])):
            raise ValueError("sign must be -1 or +1 per neuron")
        if len(np.unique(self.body)) != n:
            raise ValueError("body IDs must be unique")
        for name in ("indptr", "indices", "weight", "sign", "body"):
            getattr(self, name).flags.writeable = False

    @property
    def n(self):
        return len(self.body)

    @property
    def pre(self):
        return np.repeat(np.arange(self.n), np.diff(self.indptr))

    def subset(self, body_ids):
        """Induced circuit in requested body order; drop all boundary edges."""
        ids = _vector(body_ids, "body_ids", int)
        lookup = {int(b): i for i, b in enumerate(self.body)}
        if len(ids) == 0 or len(set(ids)) != len(ids):
            raise ValueError("select unique, nonempty body IDs")
        if any(int(b) not in lookup for b in ids):
            raise ValueError("unknown body ID")
        old = [lookup[int(b)] for b in ids]
        remap = {v: i for i, v in enumerate(old)}
        ptr, post, weight = [0], [], []
        for row in old:
            for edge in range(self.indptr[row], self.indptr[row + 1]):
                target = int(self.indices[edge])
                if target in remap:
                    post.append(remap[target])
                    weight.append(self.weight[edge])
            ptr.append(len(post))
        return Graph(np.array(ptr), np.array(post, dtype=np.int64),
                     np.array(weight), self.sign[old], ids)


def load_converted(path: str | Path, *, body_ids):
    """Load an existing flybrain NPZ, then select explicit circuit body IDs.

    No downloads/conversion/metadata inference. NPZ arrays are loaded in memory;
    supplying a small selection does not reduce peak archive-loading memory.
    """
    with np.load(path, allow_pickle=False) as data:
        required = ("indptr", "indices", "weight", "sign", "body")
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(f"missing NPZ arrays: {sorted(missing)}")
        graph = Graph(**{key: data[key] for key in required})
    return graph.subset(body_ids)


@dataclass(frozen=True)
class Parameters:
    dt: float = 1.0
    tau_state: float = 2.0
    tau_eligibility: float = 5.0
    learning_rate: float = 0.1
    weight_scale: float = 0.1
    weight_max: float = 1.0

    def __post_init__(self):
        values = vars(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("parameters must be finite")
        if any(values[k] <= 0 for k in
               ("dt", "tau_state", "tau_eligibility", "weight_max")):
            raise ValueError("time constants, dt and weight_max must be positive")
        if not np.isfinite(self.learning_rate * self.dt):
            raise ValueError("learning_rate * dt must be finite")
        if self.learning_rate < 0 or self.weight_scale < 0:
            raise ValueError("learning_rate and weight_scale must be nonnegative")


class Circuit:
    """Synchronous leaky rates and a three-factor engineering hypothesis.

    x_next = a*x + (1-a)*clip(input + signed_W*x, 0, 1)
    e_next = b*e + (1-b)*x_pre*x_next_post
    w_next = clip(w + learning_rate*dt*modulator_post*e_next, 0, max)

    a=exp(-dt/tau_state), b=exp(-dt/tau_eligibility). Only selected
    edges accumulate eligibility or learn. Signs remain fixed, and updated
    weights affect the following step. Disabling learning freezes weights but
    still updates eligibility, allowing delayed feedback after an activity pulse.
    """
    def __init__(self, graph, parameters=None, *, plastic=None):
        self.graph = graph
        self.parameters = parameters or Parameters()
        self.pre = graph.pre
        self.post = graph.indices
        if plastic is None:
            plastic = np.ones(len(self.pre), dtype=bool)
        plastic = np.asarray(plastic)
        if plastic.dtype.kind != "b" or plastic.shape != self.pre.shape:
            raise ValueError("plastic must be a boolean vector, one entry per edge")
        self.plastic = plastic.copy()
        self.state = np.zeros(graph.n)
        self.eligibility = np.zeros(len(self.pre))
        p = self.parameters
        with np.errstate(over="ignore"):
            self.weights = np.clip(graph.weight * p.weight_scale, 0, p.weight_max)
        self.time = 0.0

    def step(self, external, *, modulator=0.0, learning=True):
        external = np.asarray(external, dtype=float)
        modulator = np.asarray(modulator, dtype=float)
        if external.shape != (self.graph.n,) or not np.all(np.isfinite(external)):
            raise ValueError("external must be a finite vector, one value per neuron")
        if modulator.shape not in ((), (self.graph.n,)) or not np.all(np.isfinite(modulator)):
            raise ValueError("modulator must be a finite scalar or neuron vector")
        p = self.parameters
        with np.errstate(over="ignore", invalid="ignore"):
            learning_signal = p.learning_rate * p.dt * modulator if learning else 0.0
        if not np.all(np.isfinite(learning_signal)):
            raise ValueError("scaled modulator must be finite")
        current = np.bincount(
            self.post,
            weights=self.weights * self.graph.sign[self.pre] * self.state[self.pre],
            minlength=self.graph.n,
        )
        if not np.all(np.isfinite(current)):
            raise ValueError("recurrent current overflow; reduce weight scale/bounds")
        a = np.exp(-p.dt / p.tau_state)
        b = np.exp(-p.dt / p.tau_eligibility)
        with np.errstate(over="ignore"):
            next_state = a * self.state + (1 - a) * np.clip(external + current, 0, 1)
        self.eligibility = (b * self.eligibility + (1 - b) *
                            self.state[self.pre] * next_state[self.post]) * self.plastic
        if learning:
            local = learning_signal if modulator.ndim == 0 else learning_signal[self.post]
            with np.errstate(over="ignore"):
                self.weights = np.clip(self.weights + local * self.eligibility,
                                       0, p.weight_max)
        self.state = next_state
        self.time += p.dt
        return self.state.copy()
