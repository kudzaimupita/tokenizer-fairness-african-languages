"""Typed view of configs/experiment.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

TokenizerKind = Literal["hf", "tiktoken"]


@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str
    revision: str
    filename: str
    licence: str
    local_path: Path


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    flores: str
    group: str


@dataclass(frozen=True)
class TokenizerSpec:
    key: str
    label: str
    kind: TokenizerKind
    word_prefix: str
    repo_id: str | None = None
    revision: str | None = None
    encoding: str | None = None
    note: str | None = None

    @property
    def source(self) -> str:
        """Human-readable identifier of where the tokenizer comes from."""
        if self.kind == "tiktoken":
            return f"tiktoken:{self.encoding}"
        return f"{self.repo_id}@{(self.revision or '')[:7]}"


@dataclass(frozen=True)
class Skipped:
    repo_id: str
    reason: str


@dataclass(frozen=True)
class ScatterGroup:
    label: str
    langs: list[str]


@dataclass(frozen=True)
class AnalysisConfig:
    bootstrap_samples: int
    seed: int
    confidence: float
    context_window: int
    headline_tokenizers: list[str]
    example_languages: list[str]
    n_examples: int
    min_example_chars: int
    scatter_groups: list[ScatterGroup] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    root: Path
    dataset: DatasetConfig
    baseline: str
    languages: list[Language]
    tokenizers: list[TokenizerSpec]
    skipped: list[Skipped]
    analysis: AnalysisConfig
    results_dir: Path = field(default=Path("results"))

    def language(self, code: str) -> Language:
        for lang in self.languages:
            if lang.code == code:
                return lang
        raise KeyError(code)

    def tokenizer(self, key: str) -> TokenizerSpec:
        for spec in self.tokenizers:
            if spec.key == key:
                return spec
        raise KeyError(key)

    @property
    def data_path(self) -> Path:
        return self.root / self.dataset.local_path

    @property
    def counts_path(self) -> Path:
        return self.root / "data" / "token_counts.parquet"

    @property
    def words_path(self) -> Path:
        return self.root / "data" / "word_counts.parquet"

    @property
    def out_dir(self) -> Path:
        return self.root / self.results_dir


def _tokenizer(raw: dict[str, Any]) -> TokenizerSpec:
    kind = raw["kind"]
    if kind not in ("hf", "tiktoken"):
        raise ValueError(f"unknown tokenizer kind {kind!r} for {raw['key']}")
    if kind == "hf" and not (raw.get("repo_id") and raw.get("revision")):
        raise ValueError(f"hf tokenizer {raw['key']} needs repo_id and a pinned revision")
    if kind == "tiktoken" and not raw.get("encoding"):
        raise ValueError(f"tiktoken tokenizer {raw['key']} needs an encoding name")
    return TokenizerSpec(**raw)


def load_config(path: str | Path = "configs/experiment.yaml") -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ds = raw["dataset"]
    cfg = Config(
        root=path.parent.parent,
        dataset=DatasetConfig(
            repo_id=ds["repo_id"],
            revision=ds["revision"],
            filename=ds["filename"],
            licence=ds["licence"],
            local_path=Path(ds["local_path"]),
        ),
        baseline=raw["baseline"],
        languages=[Language(**lang) for lang in raw["languages"]],
        tokenizers=[_tokenizer(t) for t in raw["tokenizers"]],
        skipped=[Skipped(**s) for s in raw.get("skipped", [])],
        analysis=AnalysisConfig(
            **{k: v for k, v in raw["analysis"].items() if k != "scatter_groups"},
            scatter_groups=[ScatterGroup(**g) for g in raw["analysis"].get("scatter_groups", [])],
        ),
    )
    codes = [lang.code for lang in cfg.languages]
    if cfg.baseline not in codes:
        raise ValueError(f"baseline {cfg.baseline!r} is not in the language list")
    if len(set(codes)) != len(codes):
        raise ValueError("duplicate language codes")
    keys = [t.key for t in cfg.tokenizers]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate tokenizer keys")
    return cfg
