"""Build every table and figure under results/ from the measured counts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from token_tax import analysis, figures
from token_tax.config import Config
from token_tax.tokenization import load_encoder


def md_table(frame: pd.DataFrame, align: Sequence[str] | None = None) -> str:
    """Minimal GitHub-flavoured Markdown table (avoids an extra dependency)."""
    cols = [str(c) for c in frame.columns]
    align = align or ["l"] + ["r"] * (len(cols) - 1)
    rule = {"l": ":--", "r": "--:", "c": ":-:"}
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(rule[a] for a in align) + " |"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _ci(row: pd.Series, name: str, digits: int = 2) -> str:
    return (
        f"{row[name]:.{digits}f} [{row[name + '_lo']:.{digits}f}, {row[name + '_hi']:.{digits}f}]"
    )


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def morphology_examples(cfg: Config, words: pd.DataFrame) -> pd.DataFrame:
    """Long, recurring isiZulu/isiXhosa words and how each headline tokenizer splits them."""
    a = cfg.analysis
    first = a.headline_tokenizers[0]
    show = [*a.headline_tokenizers, "afroxlmr", "mt5"]
    encoders = {key: load_encoder(cfg.tokenizer(key)) for key in show}
    core = words[(words["kind"] == "core") & (words["tokenizer"] == first)]
    per_lang = max(1, a.n_examples // len(a.example_languages))
    rows = []
    for code in a.example_languages:
        cand = core[
            (core["lang"] == code)
            & (core["word"].str.len() >= a.min_example_chars)
            & (core["freq"] >= 2)
            & core["word"].str.islower()
        ].sort_values(["tokens", "freq", "word"], ascending=[False, False, True])
        for word in cand["word"].head(per_lang):
            row: dict[str, object] = {"language": cfg.language(code).name, "word": word}
            for key in show:
                spec = cfg.tokenizer(key)
                pieces = encoders[key].pieces(spec.word_prefix + word)
                row[spec.label] = f"{len(pieces)}: " + " · ".join(p.strip() or "▁" for p in pieces)
            rows.append(row)
    return pd.DataFrame(rows)


def build(cfg: Config) -> None:
    out = cfg.out_dir
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    a = cfg.analysis
    labels = {t.key: t.label for t in cfg.tokenizers}
    names = {lang.code: lang.name for lang in cfg.languages}

    counts, words, stats = analysis.load_inputs(cfg)
    canon = analysis.identical_groups(counts)
    metrics = analysis.metric_table(cfg, counts, stats)
    metrics = metrics.merge(analysis.coverage_table(words), on=["tokenizer", "lang"])
    metrics["english_equivalent_context"] = [
        analysis.english_equivalent_context(a.context_window, p) for p in metrics["parity"]
    ]
    metrics.round(4).to_csv(out / "metrics.csv", index=False)
    analysis.word_sum_check(words, counts).round(4).to_csv(
        out / "sanity_word_sums.csv", index=False
    )

    # Tokenizer inventory.
    status = pd.read_csv(out / "tokenizers.csv")
    status["identical_to"] = [
        labels[canon[k]] if canon.get(k, k) != k else "" for k in status["key"]
    ]
    status.to_csv(out / "tokenizers.csv", index=False)
    inv = status.assign(source=status["source"].map(lambda s: f"`{s}`"))
    inv = inv[["label", "source", "vocab_size", "identical_to", "status"]]
    skipped = pd.DataFrame([{"repo": f"`{s.repo_id}`", "reason": s.reason} for s in cfg.skipped])
    _write(
        out / "tokenizers.md",
        md_table(inv, ["l", "l", "r", "l", "l"]) + "\nSkipped:\n\n" + md_table(skipped, ["l", "l"]),
    )

    # Order: languages by median parity, tokenizers by mean parity on Southern African languages.
    langs = analysis.language_summary(cfg, metrics, canon).sort_values("median_parity")
    southern = [lang.code for lang in cfg.languages if lang.group == "southern_african"]
    tok_order = (
        metrics[metrics["lang"].isin(southern)]
        .groupby("tokenizer")["parity"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    matrix = metrics.pivot(index="lang", columns="tokenizer", values="parity")
    matrix = matrix.loc[langs["lang"], tok_order]
    matrix.index = [names[c] for c in matrix.index]
    matrix.columns = [labels[c] for c in matrix.columns]
    matrix.round(3).to_csv(out / "parity_matrix.csv")
    _write(
        out / "parity_matrix.md",
        md_table(
            matrix.map(lambda v: f"{v:.2f}").reset_index().rename(columns={"index": "language"})
        ),
    )

    # Absolute tokens per sentence: parity alone hides a costly English baseline.
    absolute = metrics.pivot(index="lang", columns="tokenizer", values="tokens_per_sentence")
    absolute = absolute.loc[langs["lang"], tok_order]
    absolute.index = [names[c] for c in absolute.index]
    absolute.columns = [labels[c] for c in absolute.columns]
    absolute.round(2).to_csv(out / "tokens_per_sentence.csv")
    _write(
        out / "tokens_per_sentence.md",
        md_table(
            absolute.map(lambda v: f"{v:.1f}").reset_index().rename(columns={"index": "language"})
        ),
    )

    lang_table = langs.copy()
    for col in ("median_parity", "min_parity", "max_parity", "length_ratio", "median_fertility"):
        lang_table[col] = lang_table[col].map(lambda v: f"{v:.2f}")
    lang_table["median_single_token_rate"] = langs["median_single_token_rate"].map(
        lambda v: f"{100 * v:.0f}%"
    )
    lang_table.drop(columns="lang").to_csv(out / "languages.csv", index=False)
    _write(out / "languages.md", md_table(lang_table.drop(columns="lang")))

    # Headline tokenizers: full metrics with CIs.
    lang_order = langs["lang"].tolist()
    for key in a.headline_tokenizers:
        sub = metrics[metrics["tokenizer"] == key].set_index("lang").loc[lang_order]
        tab = pd.DataFrame(
            {
                "language": [names[c] for c in sub.index],
                "tokens/sentence": [_ci(r, "tokens_per_sentence", 1) for _, r in sub.iterrows()],
                "fertility": [_ci(r, "fertility") for _, r in sub.iterrows()],
                "chars/token": [_ci(r, "chars_per_token") for _, r in sub.iterrows()],
                "bytes/token": [_ci(r, "bytes_per_token") for _, r in sub.iterrows()],
                "parity (= cost x)": [_ci(r, "parity") for _, r in sub.iterrows()],
                f"sentences in {a.context_window // 1024}k": [
                    f"{r.sentences_in_context} [{r.sentences_in_context_lo}, "
                    f"{r.sentences_in_context_hi}]"
                    for r in sub.itertuples()
                ],
                "English-token equivalent of 8k": sub["english_equivalent_context"].tolist(),
            }
        )
        _write(out / f"metrics_{key}.md", md_table(tab))

    # Decomposition: text length vs tokenizer compression.
    dec = metrics[metrics["tokenizer"].isin(a.headline_tokenizers)]
    dec_tab = dec.pivot(index="lang", columns="tokenizer", values="compression_gap")
    dec_tab = dec_tab.loc[lang_order, a.headline_tokenizers]
    length = dec.groupby("lang")["length_ratio"].first().loc[lang_order]
    dec_out = pd.DataFrame(
        {
            "language": [names[c] for c in lang_order],
            "length ratio (chars vs English)": length.round(2).to_numpy(),
        }
    )
    for key in a.headline_tokenizers:
        dec_out[f"compression gap, {labels[key]}"] = dec_tab[key].round(2).to_numpy()
    _write(out / "decomposition.md", md_table(dec_out))

    # Coverage vs parity.
    corr = analysis.correlation(cfg, metrics, canon)
    cov = metrics.pivot(index="lang", columns="tokenizer", values="single_token_rate")
    cov = (100 * cov.loc[lang_order, tok_order]).round(0).astype(int)
    cov.index = [names[c] for c in cov.index]
    cov.columns = [labels[c] for c in cov.columns]
    _write(out / "coverage.md", md_table(cov.reset_index().rename(columns={"index": "language"})))

    # Morphology examples.
    examples = morphology_examples(cfg, words)
    examples.to_csv(out / "morphology_examples.csv", index=False)
    _write(out / "morphology_examples.md", md_table(examples, ["l"] * len(examples.columns)))

    # Figures.
    distinct = [k for k in tok_order if canon[k] == k]
    heat = matrix[[labels[k] for k in distinct]].copy()
    heat.columns = [" / ".join(labels[k] for k in tok_order if canon[k] == d) for d in distinct]
    figures.parity_heatmap(
        heat,
        fig_dir / "parity_heatmap.png",
        "Tokens needed relative to English (FLORES-200 devtest)",
    )
    bars = metrics[metrics["tokenizer"].isin(a.headline_tokenizers)].assign(
        language=lambda f: f["lang"].map(names)
    )
    bar_langs = [names[c] for c in lang_order if c != cfg.baseline]
    figures.parity_bars(
        bars,
        [(k, labels[k]) for k in a.headline_tokenizers],
        bar_langs,
        fig_dir / "parity_bars.png",
        "Token tax per language, with 95% bootstrap CIs",
    )
    scatter = metrics[(metrics["lang"] != cfg.baseline) & metrics["tokenizer"].isin(distinct)]
    figures.coverage_scatter(
        scatter,
        [(g.label, g.langs) for g in a.scatter_groups],
        fig_dir / "coverage_vs_parity.png",
        "Whole-word coverage vs parity (one point per language x tokenizer)",
    )

    summary = {
        "n_sentences": int(stats["sentence_id"].nunique()),
        "bootstrap_samples": a.bootstrap_samples,
        "seed": a.seed,
        "identical_tokenizers": {k: v for k, v in canon.items() if k != v},
        "correlation": corr,
        "languages": langs.round(4).to_dict(orient="records"),
    }
    _write(out / "summary.json", json.dumps(summary, indent=2) + "\n")
