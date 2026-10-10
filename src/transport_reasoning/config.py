"""Validated, hashable run specifications."""
import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


METHODS = ("onehop", "binomial", "exponential_truncated", "finite_geometric",
           "learned_polynomial", "recurrent")
ROOT = Path("experiments/transport_reasoning")


@dataclass
class Config:
    task: str = "maze"
    method: str = "binomial"
    stage: str = "smoke"
    seed: int = 101
    data_seed: int = 20261009
    size: int = 4
    train_count: int = 64
    val_count: int = 16
    test_count: int = 16
    width: int = 32
    heads: int = 4
    layers: int = 2
    hops: int = 4
    recurrent_steps: int = 8
    support: str = "graph"
    inner_beta: float = 0.0
    inner_width: int = 0
    batch_size: int = 16
    steps: int = 10
    eval_every: int = 5
    lr: float = 0.001
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    clip: float = 1.0
    device: str = "cpu"
    threads: int = 2
    max_seconds: float = 3600
    generator: str = "dfs"
    clue_fraction: float = 0.65
    data_tag: str = "smoke-v1"
    data_source: str = "generated"
    overfit_count: int = 0

    def validate(self):
        if self.task not in ("maze", "sudoku") or self.method not in METHODS:
            raise ValueError("Unknown task or method")
        if self.data_source not in ("generated","rrn"):
            raise ValueError("Unknown data source")
        if self.data_source=="rrn" and (self.task!="sudoku" or self.size!=9):
            raise ValueError("RRN source is only for Sudoku 9x9")
        if self.support not in ("graph", "dense") or self.generator not in ("dfs", "wilson"):
            raise ValueError("Invalid support or generator")
        for key in ("size", "train_count", "val_count", "test_count", "width", "heads",
                    "layers", "hops", "recurrent_steps", "batch_size", "steps", "eval_every", "threads"):
            if getattr(self, key) <= 0: raise ValueError(f"{key} must be positive")
        if self.width % self.heads or self.width % 4:
            raise ValueError("width must be divisible by heads and four")
        if self.task == "sudoku" and self.size not in (4, 9):
            raise ValueError("Sudoku size must be 4 or 9")
        if not 0 < self.clue_fraction < 1 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("Invalid clue/warmup fraction")
        if self.lr <= 0 or self.clip <= 0 or self.max_seconds <= 0 or self.inner_beta < 0:
            raise ValueError("Invalid numeric setting")
        if self.stage == "final" and self.seed not in (11, 23, 37, 53, 71):
            raise ValueError("Final seeds must match the protocol")
        if self.method == "recurrent" and self.inner_beta:
            raise ValueError("Inner MLP series uses cached filters")
        return self

    def dict(self): return asdict(self)

    def hash(self):
        return digest(self.dict())[:16]

    def data_spec(self):
        return {k: getattr(self, k) for k in ("task", "size", "data_seed", "train_count",
                "val_count", "test_count", "generator", "clue_fraction", "data_tag", "data_source")}

    def data_dir(self): return ROOT / "data" / digest(self.data_spec())[:16]

    def run_dir(self):
        return ROOT / "runs" / self.task / self.method / self.hash() / f"seed_{self.seed}"


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path):
    d = json.loads(Path(path).read_text())
    unknown = set(d) - {f.name for f in fields(Config)}
    if unknown: raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**d).validate()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)
