"""Thai → IPA data module.

Source: Wiktionary Thai↔IPA pairs (filtered, deduplicated). Encoded at
the character level on both sides (Thai source alphabet, IPA target
alphabet). Both vocab sizes are small (< 100 chars), so the student
transformer remains compact.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from framework.data import DataModule, DataSplit, Example, PreparedData
from framework.registry import register_data_module

THAI_ALPHABET = (
    "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
    "ะาิีึืุู็่้๊๋์ๆฯ "
    "0123456789"
)

IPA_ALPHABET = (
    "ɐɑɒɓɔɕɖɘəɛɚɜɞɟʄɡɠɢɦɥɧɨɪɫɬɭɮɱɲɳɴøɵɸθœɶʘɹɺɾɻʀʁʂʃʈʈʰʉʊʋⱱʌɣɤʍχʎʏźʐʒʒʲˈˌːˑː̃ʰʷˤ"
    "aeioubcdfghjklmnpqrstvwyz "
)

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


def _build_vocab(alphabet: str) -> dict[str, int]:
    special = {"<pad>": PAD_ID, "<sos>": SOS_ID, "<eos>": EOS_ID}
    chars = {c: i + len(special) for i, c in enumerate(alphabet)}
    return {**special, **chars}


INPUT_VOCAB = _build_vocab(THAI_ALPHABET)
OUTPUT_VOCAB = _build_vocab(IPA_ALPHABET)


def clean_thai(text: str) -> str:
    out = [c for c in text if c in THAI_ALPHABET]
    cleaned = "".join(out)
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip()


def clean_ipa(text: str) -> str:
    out = [c for c in text if c in IPA_ALPHABET]
    cleaned = "".join(out)
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip()


@register_data_module("secryst_thai_ipa_data")
class SecrystThaiIpaData(DataModule):
    """Concrete data module for secryst_thai_ipa."""

    def prepare_data(self) -> PreparedData:
        if self._prepared is not None:
            return self._prepared
        raw_path = self.data_root / "raw" / f"{self.config.source}.tsv"
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
            max_seq_len=max(max(len(e.source), len(e.target)) for e in all_examples) + 1,
        )
        self._prepared = prepared
        return prepared

    def encode_source(self, text: str) -> tuple[int, ...]:
        cleaned = clean_thai(text)
        return tuple(INPUT_VOCAB.get(c, PAD_ID) for c in cleaned)

    def decode_target(self, ids: Sequence[int]) -> str:
        inv = {v: k for k, v in OUTPUT_VOCAB.items() if k not in {"<pad>", "<sos>", "<eos>"}}
        return "".join(inv.get(int(i), "") for i in ids if int(i) not in {PAD_ID, SOS_ID, EOS_ID})

    def _read_examples(self, path: Path) -> list[tuple[str, str]]:
        if not path.is_file():
            return []
        out: list[tuple[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            thai = clean_thai(parts[0])
            ipa = clean_ipa(parts[1])
            if not thai or not ipa:
                continue
            out.append((thai, ipa))
        return out

    def _fallback_examples(self) -> list[tuple[str, str]]:
        return [
            ("สวัสดี", "saː.wàt.diː"),
            ("ขอบคุณ", "kʰɔ̌ːp.kʰun"),
            ("รัก", "rák"),
            ("บ้าน", "bâːn"),
            ("น้ำ", "náːm"),
            ("อาหาร", "ʔaː.hǎːn"),
            ("เพื่อน", "pʰɯ̂a̯n"),
            ("ครอบครัว", "kʰrɔ̂ːp.kʰrua̯"),
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
