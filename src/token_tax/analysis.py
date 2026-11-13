"""Turn raw token counts into the tables reported in the README."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from token_tax.config import Config
from token_tax.metrics import (
    bootstrap_indices,
    bootstrap_mean,
    bootstrap_ratio,
    parity_decomposition,
    pearson,
    sentences_in_context,
    spearman,
)


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.read_parquet(cfg.counts_path)
    words = pd.read_parquet(cfg.words_path)
    stats = pd.read_parquet(cfg.root / "data" / "sentence_stats.parquet")
    return counts, words, stats


def identical_groups(counts: pd.DataFrame) -> dict[str, str]:
    """Map each tokenizer to the first tokenizer with exactly the same counts everywhere.

    Two tokenizers that give identical token counts for all 10,120 sentences are treated
    as one vocabulary when pooling across tokenizers (so they are not double-counted).
    """
    order = list(dict.fromkeys(counts["tokenizer"]))
    vectors = {
        key: tuple(g.sort_values(["lang", "sentence_id"])["tokens"].to_numpy())
        for key, g in counts.groupby("tokenizer", sort=False)
    }
    canon: dict[str, str] = {}
    for key in order:
        canon[key] = next(k for k in order if vectors[k] == vectors[key])
    return canon


def metric_table(cfg: Config, counts: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Per (tokenizer, language) point estimates with paired bootstrap 95% CIs."""
    a = cfg.analysis
    n = stats["sentence_id"].nunique()
    idx = bootstrap_indices(n, a.bootstrap_samples, a.seed)
    size = {lang: g.sort_values("sentence_id") for lang, g in stats.groupby("lang")}
    base_size = size[cfg.baseline]

    rows = []
    for tok, tg in counts.groupby("tokenizer", sort=False):
        per_lang = {lang: g.sort_values("sentence_id") for lang, g in tg.groupby("lang")}
        base = per_lang[cfg.baseline]["tokens"].to_numpy()
        for lang in cfg.languages:
            t = per_lang[lang.code]["tokens"].to_numpy()
            s = size[lang.code]
            chars, nbytes, words = (s[c].to_numpy() for c in ("chars", "bytes", "words"))
            est = {
                "tokens_per_sentence": bootstrap_mean(t, idx, a.confidence),
                "fertility": bootstrap_ratio(t, words, idx, a.confidence),
                "chars_per_token": bootstrap_ratio(chars, t, idx, a.confidence),
                "bytes_per_token": bootstrap_ratio(nbytes, t, idx, a.confidence),
                "parity": bootstrap_ratio(t, base, idx, a.confidence),
            }
            row: dict[str, object] = {"tokenizer": tok, "lang": lang.code}
            for name, e in est.items():
                row[name] = e.value
                row[f"{name}_lo"] = e.low
                row[f"{name}_hi"] = e.high
            tps = est["tokens_per_sentence"]
            row["sentences_in_context"] = sentences_in_context(a.context_window, tps.value)
            row["sentences_in_context_lo"] = sentences_in_context(a.context_window, tps.high)
            row["sentences_in_context_hi"] = sentences_in_context(a.context_window, tps.low)
            # At an equal per-token price the cost multiplier is the parity itself.
            row["cost_multiplier"] = est["parity"].value
            _, length_ratio, gap = parity_decomposition(
                t.sum(), chars.sum(), base.sum(), base_size["chars"].to_numpy().sum()
            )
            row["length_ratio"] = length_ratio
            row["compression_gap"] = gap
            row["unk_tokens"] = int(per_lang[lang.code]["unk"].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def coverage_table(words: pd.DataFrame) -> pd.DataFrame:
    """Share of word occurrences that the tokenizer keeps as a single token."""
    core = words[words["kind"] == "core"].copy()
    core["single"] = (core["tokens"] == 1) * core["freq"]
    core["weighted_tokens"] = core["tokens"] * core["freq"]
    grouped = core.groupby(["tokenizer", "lang"]).agg(
        single=("single", "sum"), freq=("freq", "sum"), wt=("weighted_tokens", "sum")
    )
    out = pd.DataFrame(
        {
            "single_token_rate": grouped["single"] / grouped["freq"],
            "tokens_per_word_isolated": grouped["wt"] / grouped["freq"],
        }
    )
    return out.reset_index()


def word_sum_check(words: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Sum of isolated-word token counts divided by sentence token counts (should be ~1)."""
    raw = words[words["kind"] == "raw"]
    word_sum = (raw["tokens"] * raw["freq"]).groupby([raw["tokenizer"], raw["lang"]]).sum()
    sent_sum = counts.groupby(["tokenizer", "lang"])["tokens"].sum()
    return (word_sum / sent_sum).rename("word_sum_ratio").reset_index()


def correlation(cfg: Config, metrics: pd.DataFrame, canon: dict[str, str]) -> dict[str, object]:
    """Correlate parity with single-token coverage over (language, distinct tokenizer)."""
    distinct = sorted({v for v in canon.values()})
    sub = metrics[(metrics["lang"] != cfg.baseline) & metrics["tokenizer"].isin(distinct)]
    log_parity = np.log(sub["parity"].to_numpy())
    cover = sub["single_token_rate"].to_numpy()
    per_tok = [spearman(g["single_token_rate"], g["parity"]) for _, g in sub.groupby("tokenizer")]
    # Within one language, across tokenizers: holds orthography and text fixed.
    per_lang = {
        str(lang): spearman(g["single_token_rate"], g["parity"]) for lang, g in sub.groupby("lang")
    }
    return {
        "n_points": len(sub),
        "n_tokenizers": len(distinct),
        "spearman_pooled": spearman(cover, log_parity),
        "pearson_pooled_log_parity": pearson(cover, log_parity),
        "spearman_within_tokenizer_median": float(np.median(per_tok)),
        "spearman_within_tokenizer_min": float(np.min(per_tok)),
        "spearman_within_tokenizer_max": float(np.max(per_tok)),
        "spearman_within_language_median": float(np.median(list(per_lang.values()))),
        "spearman_within_language_min": float(np.min(list(per_lang.values()))),
        "spearman_within_language_max": float(np.max(list(per_lang.values()))),
        "spearman_within_language": per_lang,
    }


def language_summary(cfg: Config, metrics: pd.DataFrame, canon: dict[str, str]) -> pd.DataFrame:
    """Per language: spread of parity across distinct tokenizers and the best one."""
    distinct = sorted(set(canon.values()))
    labels = {t.key: t.label for t in cfg.tokenizers}
    rows = []
    for lang in cfg.languages:
        g = metrics[(metrics["lang"] == lang.code) & metrics["tokenizer"].isin(distinct)]
        ranked = g.sort_values("parity", kind="mergesort")
        best = ranked.iloc[0]
        worst = ranked.iloc[-1]
        is_base = lang.code == cfg.baseline  # parity is 1.0 for every tokenizer
        rows.append(
            {
                "lang": lang.code,
                "language": lang.name,
                "median_parity": float(g["parity"].median()),
                "min_parity": float(best["parity"]),
                "best_tokenizer": "-" if is_base else labels[str(best["tokenizer"])],
                "max_parity": float(worst["parity"]),
                "worst_tokenizer": "-" if is_base else labels[str(worst["tokenizer"])],
                "length_ratio": float(g["length_ratio"].iloc[0]),
                "median_fertility": float(g["fertility"].median()),
                "median_single_token_rate": float(g["single_token_rate"].median()),
            }
        )
    return pd.DataFrame(rows)


def english_equivalent_context(context_window: int, parity: float) -> int:
    """English-token budget that an 8k window holds when filled with this language."""
    return math.floor(context_window / parity)
