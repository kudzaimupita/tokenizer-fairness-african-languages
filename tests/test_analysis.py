from pathlib import Path

import pandas as pd
import pytest

from token_tax.analysis import coverage_table, english_equivalent_context, identical_groups
from token_tax.config import load_config
from token_tax.report import md_table

ROOT = Path(__file__).resolve().parents[1]


def test_project_config_loads_and_is_pinned() -> None:
    cfg = load_config(ROOT / "configs" / "experiment.yaml")
    assert cfg.baseline == "eng"
    assert {lang.code for lang in cfg.languages} >= {"zul", "xho", "sot", "tsn", "sna", "swh"}
    for spec in cfg.tokenizers:
        if spec.kind == "hf":
            assert spec.revision is not None and len(spec.revision) == 40
    assert set(cfg.analysis.headline_tokenizers) <= {t.key for t in cfg.tokenizers}
    grouped = {code for g in cfg.analysis.scatter_groups for code in g.langs}
    assert grouped == {lang.code for lang in cfg.languages} - {cfg.baseline}


def test_identical_groups_merges_exact_duplicates() -> None:
    rows = []
    for tok, counts in {"a": [3, 4], "b": [3, 4], "c": [3, 5]}.items():
        for sid, n in enumerate(counts):
            rows.append({"tokenizer": tok, "lang": "eng", "sentence_id": sid, "tokens": n})
    canon = identical_groups(pd.DataFrame(rows))
    assert canon == {"a": "a", "b": "a", "c": "c"}


def test_coverage_table_weights_by_frequency() -> None:
    words = pd.DataFrame(
        {
            "tokenizer": ["t"] * 3,
            "lang": ["zul"] * 3,
            "kind": ["core"] * 3,
            "word": ["a", "bb", "ccc"],
            "freq": [8, 1, 1],
            "tokens": [1, 2, 4],
        }
    )
    row = coverage_table(words).iloc[0]
    assert row["single_token_rate"] == pytest.approx(0.8)
    assert row["tokens_per_word_isolated"] == pytest.approx((8 + 2 + 4) / 10)


def test_english_equivalent_context() -> None:
    assert english_equivalent_context(8192, 1.0) == 8192
    assert english_equivalent_context(8192, 2.0) == 4096


def test_md_table() -> None:
    frame = pd.DataFrame({"language": ["isiZulu"], "parity": ["1.72"]})
    assert md_table(frame) == "| language | parity |\n| :-- | --: |\n| isiZulu | 1.72 |\n"
