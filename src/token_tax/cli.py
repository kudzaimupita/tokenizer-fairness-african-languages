"""Command line entry point: `token-tax {data,measure,report,all}`."""

from __future__ import annotations

import argparse

from token_tax.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="token-tax", description=__doc__)
    parser.add_argument("step", choices=["data", "measure", "report", "all"])
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.step in ("data", "all"):
        from token_tax.data import download

        download(cfg)
    if args.step in ("measure", "all"):
        from token_tax.measure import run

        run(cfg)
    if args.step in ("report", "all"):
        from token_tax.report import build

        build(cfg)


if __name__ == "__main__":
    main()
