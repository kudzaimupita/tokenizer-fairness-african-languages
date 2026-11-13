.PHONY: all data measure report test lint typecheck check clean

export TRANSFORMERS_VERBOSITY := error

UV := uv run

all: data measure report

data:  ## download FLORES-200 devtest (CC-BY-SA 4.0) into data/
	$(UV) token-tax data

measure:  ## tokenise every sentence and word with every tokenizer
	$(UV) token-tax measure

report:  ## write tables, figures and summary.json under results/
	$(UV) token-tax report

test:
	$(UV) pytest

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

typecheck:
	$(UV) mypy

check: lint typecheck test

clean:
	rm -rf data/
