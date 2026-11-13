"""A thin common interface over Hugging Face and tiktoken tokenizers.

Only tokenizer files are downloaded (tokenizer.json, sentencepiece models, vocab and
merges, plus the small config.json that AutoTokenizer reads). No model weights.
Special tokens (BOS/EOS/CLS/SEP) are never added: we count the text itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from token_tax.config import TokenizerSpec


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...

    def pieces(self, text: str) -> list[str]: ...

    def unk_id(self) -> int | None: ...

    def vocab_size(self) -> int: ...


@dataclass
class HFEncoder:
    spec: TokenizerSpec

    def __post_init__(self) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(
            self.spec.repo_id,
            revision=self.spec.revision,
            token=False,  # anonymous: gated repos fail here instead of using a stored login
        )

    def encode(self, text: str) -> list[int]:
        ids: list[int] = self._tok.encode(text, add_special_tokens=False)
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        batch: list[list[int]] = self._tok(texts, add_special_tokens=False)["input_ids"]
        return batch

    def pieces(self, text: str) -> list[str]:
        """Token strings made readable: each id decoded on its own."""
        return [str(self._tok.decode([i])) for i in self.encode(text)]

    def unk_id(self) -> int | None:
        unk: int | None = self._tok.unk_token_id
        return unk

    def vocab_size(self) -> int:
        return len(self._tok)


@dataclass
class TiktokenEncoder:
    spec: TokenizerSpec

    def __post_init__(self) -> None:
        import tiktoken

        assert self.spec.encoding is not None
        self._enc = tiktoken.get_encoding(self.spec.encoding)

    def encode(self, text: str) -> list[int]:
        # disallowed_special=() treats text like "<|endoftext|>" as plain text.
        return self._enc.encode(text, disallowed_special=())

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self._enc.encode_batch(texts, disallowed_special=())

    def pieces(self, text: str) -> list[str]:
        return [
            self._enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
            for i in self.encode(text)
        ]

    def unk_id(self) -> int | None:
        return None  # byte-level BPE has no unknown token

    def vocab_size(self) -> int:
        return int(self._enc.n_vocab)


def load_encoder(spec: TokenizerSpec) -> Encoder:
    if spec.kind == "tiktoken":
        return TiktokenEncoder(spec)
    return HFEncoder(spec)
