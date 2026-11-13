"""Download FLORES-200 devtest and keep only the languages in the config."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from token_tax.config import Config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(cfg: Config) -> Path:
    """Fetch the pinned parquet file anonymously and write a small per-language copy."""
    ds = cfg.dataset
    raw = Path(
        hf_hub_download(
            repo_id=ds.repo_id,
            filename=ds.filename,
            revision=ds.revision,
            repo_type="dataset",
            token=False,  # anonymous on purpose: the study must be reproducible without an account
        )
    )
    frame = pd.read_parquet(raw)
    columns = {lang.flores: lang.code for lang in cfg.languages}
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"languages missing from {ds.repo_id}: {missing}")
    subset = frame[list(columns)].rename(columns=columns).reset_index(drop=True)
    validate(subset)
    subset.insert(0, "sentence_id", range(len(subset)))

    out = cfg.data_path
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(out, index=False)
    print(f"{ds.repo_id}@{ds.revision[:7]} {ds.filename} sha256={_sha256(raw)}")
    print(f"wrote {len(subset)} aligned sentences x {len(columns)} languages -> {out}")
    return out


def validate(frame: pd.DataFrame) -> None:
    """Fail loudly on empty or non-string cells; parity needs every sentence present."""
    for column in frame.columns:
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"{column}: {int(values.isna().sum())} missing sentences")
        empty = values.map(lambda s: not isinstance(s, str) or not s.strip())
        if empty.any():
            raise ValueError(f"{column}: {int(empty.sum())} empty sentences")


def load_sentences(cfg: Config) -> pd.DataFrame:
    path = cfg.data_path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make data` first")
    frame = pd.read_parquet(path)
    validate(frame.drop(columns=["sentence_id"]))
    return frame
