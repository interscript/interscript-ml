"""Model.load(zip) + translate(text): greedy KV decode with plain fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from interscript_ml.loader import load_manifest, verify_and_read
from interscript_ml.tokens import EOS_ID, PAD_ID, decode, encode


class Model:
    """A loaded, checksum-verified IMF v1 model.

    >>> model = Model.load("khm-latn-1.0.zip")
    >>> model.translate("ភាសា")
    """

    def __init__(self, zip_path: Path | str):
        self.zip_path = Path(zip_path)
        self.manifest = load_manifest(self.zip_path)
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        graphs = verify_and_read(self.zip_path)
        self._encoder = ort.InferenceSession(
            graphs["encoder.onnx"], options, providers=_providers()
        )
        decoder_name = (
            "decoder-kv.onnx"
            if self.manifest.decoder == "kv" and "decoder-kv.onnx" in graphs
            else "decoder.onnx"
        )
        self._kv_session = decoder_name == "decoder-kv.onnx"
        self._decoder = ort.InferenceSession(
            graphs[decoder_name], options, providers=_providers()
        )
        self._pasts = {
            meta.name: _zero_past(meta)
            for meta in self._decoder.get_inputs()
            if meta.name.startswith("past_")
        }
        self._output_names = [o.name for o in self._decoder.get_outputs()]

    @classmethod
    def load(cls, path_or_id: Path | str, index_url: str | None = None) -> Model:
        """Accepts a zip path OR a model id from models.yaml (dynamic
        fetch: download -> verify -> cache)."""
        candidate = str(path_or_id)
        if candidate.endswith(".zip") or Path(candidate).exists():
            return cls(candidate)
        from interscript_ml.registry import resolve

        return cls(resolve(candidate, index_url))

    @property
    def id(self) -> str:
        return self.manifest.id

    def translate(self, text: str, max_len: int = 256, num_beams: int = 1) -> str:
        token_ids = self.generate(text, max_len=max_len, num_beams=num_beams)
        return decode(token_ids)

    def generate(self, text: str, max_len: int = 256, num_beams: int = 1) -> list[int]:
        ids = np.array([encode(text)], dtype=np.int64)
        if ids.shape[1] == 1:  # only the trailing EOS: empty input
            return []
        hidden = self._encoder.run(None, {"input_ids": ids})[0]
        if self._kv_session:
            if num_beams > 1:
                return self._beam_kv(hidden, max_len, num_beams)
            return self._greedy_kv(hidden, max_len)
        return self._greedy_plain(hidden, max_len)

    def _beam_kv(self, hidden, max_len: int, num_beams: int) -> list[int]:
        """Batched beam search over the KV graph: the export's batch axis
        carries the beams; per-step presents are gathered on beam reorder.
        Canonical semantics: EOS hypotheses are recorded but never shrink
        the live set (candidates come from a 2K window); the search runs
        to max_len or exhaustion, and the winner is picked by raw
        cumulative logprob — length normalization measurably rewards
        long garbage on low-confidence byte models."""
        beams = num_beams
        enc = np.repeat(hidden, beams, axis=0)
        pasts = {
            name: np.repeat(zero, beams, axis=0)
            for name, zero in self._pasts.items()
        }
        current = np.full((beams, 1), PAD_ID, dtype=np.int64)
        scores = np.full((beams,), -np.inf, dtype=np.float32)
        scores[0] = 0.0  # only beam 0 is live at step 0
        sequences: list[list[int]] = [[] for _ in range(beams)]
        finished: list[tuple[float, list[int]]] = []

        for _ in range(max_len):
            outputs = self._decoder.run(
                None,
                {"input_ids": current, "encoder_hidden_states": enc, **pasts},
            )
            results = dict(zip(self._output_names, outputs, strict=True))
            logits = results["logits"][:, -1, :].astype(np.float32)
            logprobs = logits - np.log(
                np.exp(logits - logits.max(axis=1, keepdims=True)).sum(
                    axis=1, keepdims=True
                )
            )  # stable log-softmax
            cand = np.where(
                (scores > -np.inf)[:, None],
                scores[:, None] + logprobs,
                -np.inf,
            )
            flat = cand.reshape(-1)
            window = min(2 * beams, flat.size)
            order = np.argpartition(flat, -window)[-window:]
            order = order[np.argsort(-flat[order])]
            new_pasts = {
                name: results[name.replace("past_", "present_", 1)] for name in pasts
            }
            next_pasts: dict[str, np.ndarray] = {}
            next_sequences: list[list[int]] = []
            next_scores = np.full((beams,), -np.inf, dtype=np.float32)
            next_current = np.full((beams, 1), PAD_ID, dtype=np.int64)
            slot = 0
            for idx in order:
                src = int(idx // cand.shape[1])
                token = int(idx % cand.shape[1])
                score = float(flat[idx])
                if token == EOS_ID:
                    finished.append((score, sequences[src]))
                    continue
                if slot >= beams:
                    continue
                next_scores[slot] = score
                next_sequences.append(sequences[src] + [token])
                next_current[slot, 0] = token
                for name, tensor in new_pasts.items():
                    holder = next_pasts.setdefault(
                        name, np.empty((beams,) + tensor.shape[1:], tensor.dtype)
                    )
                    holder[slot : slot + 1] = tensor[src : src + 1]
                slot += 1
            if slot == 0:
                break
            pasts = next_pasts
            sequences = next_sequences + [
                [] for _ in range(beams - len(next_sequences))
            ]
            scores = next_scores
            current = next_current

        pool = finished + [
            (float(scores[i]), sequences[i]) for i in range(beams) if scores[i] > -np.inf
        ]
        best = max(pool, key=lambda s: s[0])
        return best[1]

    def _greedy_kv(self, hidden, max_len: int) -> list[int]:
        pasts = dict(self._pasts)
        current = np.array([[PAD_ID]], dtype=np.int64)
        generated: list[int] = []
        for _ in range(max_len):
            outputs = self._decoder.run(
                None,
                {"input_ids": current, "encoder_hidden_states": hidden, **pasts},
            )
            results = dict(zip(self._output_names, outputs, strict=True))
            token = int(np.argmax(results["logits"][0, -1]))
            if token == EOS_ID:
                break
            generated.append(token)
            pasts = {
                name: results[name.replace("past_", "present_", 1)]
                for name in pasts
            }
            current = np.array([[token]], dtype=np.int64)
        return generated

    def _greedy_plain(self, hidden, max_len: int) -> list[int]:
        decoder_ids = np.array([[PAD_ID]], dtype=np.int64)
        generated: list[int] = []
        for _ in range(max_len):
            logits = self._decoder.run(
                None,
                {"input_ids": decoder_ids, "encoder_hidden_states": hidden},
            )[0]
            token = int(np.argmax(logits[0, -1]))
            if token == EOS_ID:
                break
            generated.append(token)
            decoder_ids = np.concatenate(
                [decoder_ids, np.array([[token]], dtype=np.int64)], axis=1
            )
        return generated


def _providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    preferred = [p for p in ("CPUExecutionProvider",) if p in available]
    return preferred or available


def _zero_past(meta) -> object:
    shape = meta.shape  # [batch, heads, past_seq, d_kv], dynamic dims are str
    heads = shape[1] if isinstance(shape[1], int) else 4
    d_kv = shape[3] if isinstance(shape[3], int) else 8
    dtype = np.float16 if meta.type == "tensor(float16)" else np.float32
    return np.zeros((1, heads, 0, d_kv), dtype=dtype)
