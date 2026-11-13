import math

import numpy as np
import pytest

from token_tax.metrics import (
    bootstrap_indices,
    bootstrap_mean,
    bootstrap_ratio,
    core_word,
    parity_decomposition,
    pearson,
    ratio_of_sums,
    sentences_in_context,
    spearman,
    utf8_len,
    whitespace_words,
)


def test_whitespace_words_keeps_punctuation_and_collapses_runs() -> None:
    assert whitespace_words('"Manje  sinamagundane,"\twanezela.') == [
        '"Manje',
        'sinamagundane,"',
        "wanezela.",
    ]
    assert whitespace_words("   ") == []


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("mice,", "mice"),
        ('"Manje', "Manje"),
        ("ngu-4", "ngu-4"),  # inner hyphen and digit are kept
        ("(1990)", ""),  # no letters left
        ("--", ""),
        ("l'eau.", "l'eau"),
    ],
)
def test_core_word(word: str, expected: str) -> None:
    assert core_word(word) == expected


def test_utf8_len_counts_bytes_not_characters() -> None:
    assert utf8_len("abc") == 3
    assert utf8_len("é") == 2
    assert utf8_len("ŋ") == 2


def test_ratio_of_sums_is_not_mean_of_ratios() -> None:
    tokens = [10, 2]
    words = [5, 1]
    assert ratio_of_sums(tokens, words) == pytest.approx(2.0)
    tokens = [10, 1]
    words = [1, 10]
    # mean of ratios would be 5.05; ratio of sums is 1.0
    assert ratio_of_sums(tokens, words) == pytest.approx(1.0)


def test_ratio_of_sums_errors() -> None:
    with pytest.raises(ValueError):
        ratio_of_sums([1, 2], [1])
    with pytest.raises(ZeroDivisionError):
        ratio_of_sums([1], [0])


def test_bootstrap_indices_are_deterministic_and_in_range() -> None:
    a = bootstrap_indices(50, 100, seed=7)
    b = bootstrap_indices(50, 100, seed=7)
    c = bootstrap_indices(50, 100, seed=8)
    assert a.shape == (100, 50)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.min() >= 0 and a.max() < 50


def test_bootstrap_ratio_point_and_interval() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(10, 40, size=400)
    lang = base * 2  # exactly twice as many tokens on every sentence
    idx = bootstrap_indices(400, 500, seed=1)
    est = bootstrap_ratio(lang, base, idx)
    assert est.value == pytest.approx(2.0)
    assert est.low == pytest.approx(2.0)
    assert est.high == pytest.approx(2.0)

    noisy = base * 2 + rng.integers(-5, 6, size=400)
    est = bootstrap_ratio(noisy, base, idx)
    assert est.low <= est.value <= est.high
    assert est.high - est.low > 0


def test_bootstrap_interval_narrows_with_more_sentences() -> None:
    rng = np.random.default_rng(3)
    widths = []
    for n in (50, 2000):
        base = rng.integers(10, 40, size=n)
        lang = base * 1.7 + rng.normal(0, 4, size=n)
        est = bootstrap_ratio(lang, base, bootstrap_indices(n, 1000, seed=2))
        widths.append(est.high - est.low)
    assert widths[1] < widths[0]


def test_bootstrap_rejects_bad_inputs() -> None:
    idx = bootstrap_indices(10, 5, seed=0)
    with pytest.raises(ValueError):
        bootstrap_ratio(np.ones(5), np.ones(5), idx)  # indices exceed data length
    with pytest.raises(ValueError):
        bootstrap_ratio(np.ones(10), np.ones(10), idx, confidence=1.0)


def test_bootstrap_mean() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    est = bootstrap_mean(values, bootstrap_indices(4, 200, seed=0))
    assert est.value == pytest.approx(2.5)
    assert 1.0 <= est.low <= 2.5 <= est.high <= 4.0
    assert est.fmt(1) == f"2.5 [{est.low:.1f}, {est.high:.1f}]"


def test_sentences_in_context() -> None:
    assert sentences_in_context(8192, 26.6) == 307
    assert sentences_in_context(8192, 8192) == 1
    assert sentences_in_context(8192, 10000) == 0
    with pytest.raises(ValueError):
        sentences_in_context(8192, 0)


def test_parity_decomposition_identity() -> None:
    parity, length_ratio, gap = parity_decomposition(
        tokens=450, chars=1200, base_tokens=250, base_chars=1000
    )
    assert parity == pytest.approx(1.8)
    assert length_ratio == pytest.approx(1.2)
    assert gap == pytest.approx(1.5)
    assert length_ratio * gap == pytest.approx(parity)


def test_pearson_and_spearman_known_values() -> None:
    x = [1, 2, 3, 4, 5]
    assert pearson(x, [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert pearson(x, [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    # monotone but non-linear: Spearman is exactly 1, Pearson is below 1
    y = [1, 4, 9, 16, 1000]
    assert spearman(x, y) == pytest.approx(1.0)
    assert pearson(x, y) < 1.0


def test_spearman_with_ties_matches_reference() -> None:
    # Average ranks [1, 2.5, 2.5, 4] vs [1, 3, 2, 4]: 4.5 / sqrt(4.5 * 5).
    assert spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(0.9486832980505138)


def test_pearson_constant_input_is_nan() -> None:
    assert math.isnan(pearson([1, 1, 1], [1, 2, 3]))
