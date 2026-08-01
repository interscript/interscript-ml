"""Hebrew nikud data module.

Same shape as ``RababaArabicData`` but with a Hebrew alphabet + nikud
point set. The framework doesn't care about the script — it sees only
``Example`` objects with token IDs.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

from framework.config import DataConfig
from framework.data import DataModule, DataSplit, Example, PreparedData
from framework.registry import register_data_module

NIKUD_POINTS = "ְֱֲֳִֵֶַָֹֺֻּֽ־ֿ׀ׁׂׅׄ"
HEBREW_ALPHABET = "אבגדהוזחטיכלמנסעפצקרשת "

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


def _build_vocab(alphabet: str) -> dict[str, int]:
    special = {"<pad>": PAD_ID, "<sos>": SOS_ID, "<eos>": EOS_ID}
    chars = {c: i + len(special) for i, c in enumerate(alphabet)}
    return {**special, **chars}


INPUT_VOCAB = _build_vocab(HEBREW_ALPHABET)
OUTPUT_VOCAB = _build_vocab(HEBREW_ALPHABET + NIKUD_POINTS)


def strip_nikud(text: str) -> str:
    return "".join(c for c in text if c not in NIKUD_POINTS)


def clean_hebrew(text: str) -> str:
    out = [c for c in text if c in HEBREW_ALPHABET or c in NIKUD_POINTS]
    cleaned = "".join(out)
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip()


@register_data_module("rababa_hebrew_data")
class RababaHebrewData(DataModule):
    """Concrete data module for rababa_hebrew. Mirrors ``RababaArabicData``."""

    def prepare_data(self) -> PreparedData:
        if self._prepared is not None:
            return self._prepared
        raw_path = self.data_root / "raw" / f"{self.config.source}.txt"
        examples = self._read_examples(raw_path)
        if not examples:
            examples = self._fallback_examples()
        random.Random(42).shuffle(examples)
        val_n = max(1, min(len(examples) // 10, self.config.max_val_samples or 1000))
        val_pairs = examples[:val_n]
        train_pairs = examples[val_n:]
        if self.config.max_train_samples:
            train_pairs = train_pairs[: self.config.max_train_samples]
        train_split = self._encode_split(train_pairs)
        val_split = self._encode_split(val_pairs)
        all_examples = list(train_split) + list(val_split)
        prepared = PreparedData(
            train=train_split,
            val=val_split,
            vocab_size=len(OUTPUT_VOCAB),
            max_seq_len=max(len(e.source) for e in all_examples) + 1,
        )
        self._prepared = prepared
        return prepared

    def encode_source(self, text: str) -> tuple[int, ...]:
        cleaned = clean_hebrew(text)
        return tuple(INPUT_VOCAB.get(c, PAD_ID) for c in cleaned)

    def decode_target(self, ids: Sequence[int]) -> str:
        inv = {v: k for k, v in OUTPUT_VOCAB.items() if k not in {"<pad>", "<sos>", "<eos>"}}
        return "".join(inv.get(int(i), "") for i in ids if int(i) not in {PAD_ID, SOS_ID, EOS_ID})

    def _read_examples(self, path: Path) -> list[tuple[str, str]]:
        if not path.is_file():
            return []
        out: list[tuple[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            diacritized = clean_hebrew(line.strip())
            if not diacritized:
                continue
            bare = strip_nikud(diacritized)
            if not bare.strip():
                continue
            out.append((bare, diacritized))
        return out

    def _fallback_examples(self) -> list[tuple[str, str]]:
        return [
            ("בראשית", "בְּרֵאשִׁית"),
            ("אלהים", "אֱלֹהִים"),
            ("שלום", "שָׁלוֹם"),
            ("תורה", "תּוֹרָה"),
            ("אהבה", "אַהֲבָה"),
        ]

    def _encode_split(self, pairs: list[tuple[str, str]]) -> DataSplit:
        examples = [
            Example(
                source=source,
                target=target,
                input_ids=tuple(INPUT_VOCAB.get(c, PAD_ID) for c in source),
                target_ids=(SOS_ID,)
                + tuple(OUTPUT_VOCAB.get(c, PAD_ID) for c in target)
                + (EOS_ID,),
            )
            for source, target in pairs
        ]
        return DataSplit(tuple(examples))


def vocab_stats() -> dict[str, int]:
    return {"input": len(INPUT_VOCAB), "output": len(OUTPUT_VOCAB)}
