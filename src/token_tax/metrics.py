"""Pure metric functions. No I/O and no tokenizer dependencies, so they are easy to test.

Conventions
-----------
All corpus-level rates are *ratios of sums* over the same set of sentences, e.g.
fertility = (total tokens) / (total words), not the mean of per-sentence ratios.
Ratios of sums are not dominated by short sentences and are what a user is billed on.
Confidence intervals come from a paired percentile bootstrap over sentences: one set
of resampled sentence indices is drawn once and reused for every language and
tokenizer, so a parity ratio compares the same resampled sentences in both languages.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def whitespace_words(text: str) -> list[str]:
    """Split on any run of whitespace. Punctuation stays attached to words."""
    return text.split()


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def core_word(word: str) -> str:
    """Strip leading/trailing punctuation and symbols; return "" if no letter remains.

    Used for the vocabulary-coverage analysis, where "mice," and "mice" should count
    as the same word and pure numbers or dashes should be ignored.
    """
    start, end = 0, len(word)
    while start < end and unicodedata.category(word[start])[0] in "PS":
        start += 1
    while end > start and unicodedata.category(word[end - 1])[0] in "PS":
        end -= 1
    core = word[start:end]
    if not any(unicodedata.category(ch).startswith("L") for ch in core):
        return ""
    return core


def ratio_of_sums(numerator: ArrayLike, denominator: ArrayLike) -> float:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    if num.shape != den.shape:
        raise ValueError(f"shape mismatch: {num.shape} vs {den.shape}")
    total = den.sum()
    if total == 0:
        raise ZeroDivisionError("denominator sums to zero")
    return float(num.sum() / total)


def bootstrap_indices(n: int, n_samples: int, seed: int) -> NDArray[np.int64]:
    """Draw `n_samples` resamples (with replacement) of `n` indices, deterministically."""
    if n <= 0 or n_samples <= 0:
        raise ValueError("n and n_samples must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_samples, n), dtype=np.int64)


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float

    def fmt(self, digits: int = 2) -> str:
        return f"{self.value:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def bootstrap_ratio(
    numerator: ArrayLike,
    denominator: ArrayLike,
    indices: NDArray[np.int64],
    confidence: float = 0.95,
) -> Estimate:
    """Ratio of sums with a paired percentile bootstrap CI.

    `numerator[i]` and `denominator[i]` must refer to the same sentence i.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    point = ratio_of_sums(num, den)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if indices.max(initial=0) >= len(num):
        raise ValueError("bootstrap indices out of range for the data")
    boot = num[indices].sum(axis=1) / den[indices].sum(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    return Estimate(point, float(low), float(high))


def bootstrap_mean(
    values: ArrayLike, indices: NDArray[np.int64], confidence: float = 0.95
) -> Estimate:
    """Mean with a percentile bootstrap CI (a ratio of sums over a vector of ones)."""
    arr = np.asarray(values, dtype=np.float64)
    return bootstrap_ratio(arr, np.ones_like(arr), indices, confidence)


def sentences_in_context(context_window: int, tokens_per_sentence: float) -> int:
    """How many average-length sentences fit into a context window."""
    if tokens_per_sentence <= 0:
        raise ValueError("tokens_per_sentence must be positive")
    return math.floor(context_window / tokens_per_sentence)


def parity_decomposition(
    tokens: float, chars: float, base_tokens: float, base_chars: float
) -> tuple[float, float, float]:
    """Split parity into a text-length factor and a compression factor.

    parity = tokens / base_tokens
           = (chars / base_chars) * ((base_chars / base_tokens) / (chars / tokens))
           = length_ratio         * compression_gap

    `length_ratio` says how much longer the translation is in characters;
    `compression_gap` says how much worse the tokenizer packs characters into tokens.
    Returns (parity, length_ratio, compression_gap); the product identity holds exactly.
    """
    parity = tokens / base_tokens
    length_ratio = chars / base_chars
    compression_gap = (base_chars / base_tokens) / (chars / tokens)
    return parity, length_ratio, compression_gap


def _ranks(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average ranks (1-based), ties share the mean of their positions."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x: ArrayLike, y: ArrayLike) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("need two equal-length vectors with at least 2 points")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum() * (b * b).sum()))
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    return pearson(_ranks(a), _ranks(b))
