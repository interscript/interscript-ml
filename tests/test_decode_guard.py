"""Repetition guard in the greedy KV decode: flat-byte students loop
on short inputs (live: كتاب -> كَتَابٍ: كَتَابٍ: ... until max_len).
The guard must cut generation when the recent token window cycles."""

from src.imf.export import onnx_greedy_kv


class _FakeEncoder:
    def run(self, _, feeds):
        import numpy as np

        return [np.zeros((1, feeds["input_ids"].shape[1], 4), dtype=np.float32)]


class _FakeKV:
    """Cycles two tokens forever, never EOS — the pathological loop."""

    def __init__(self):
        import numpy as np

        self.np = np
        self.step = 0

    def get_outputs(self):
        class O:
            def __init__(self, name):
                self.name = name

        return [O("logits"), O("present_k"), O("present_v")]

    def get_inputs(self):
        class I:
            def __init__(self, name, shape, typ="tensor(float)"):
                self.name = name
                self.shape = shape
                self.type = typ

        return [
            I("input_ids", [1, "seq"]),
            I("encoder_hidden_states", [1, "seq", 4]),
            I("past_k", [1, 4, "past", 8]),
            I("past_v", [1, 4, "past", 8]),
        ]

    def run(self, _, feeds):
        import numpy as np

        logits = np.full((1, 1, 260), -1e9)
        # visible cycle: token 10, 11, 10, 11, ...
        logits[0, -1, 10 if self.step % 2 == 0 else 11] = 1e9
        self.step += 1
        return [logits, np.zeros((1,)), np.zeros((1,))]


def test_looping_model_is_cut_before_max_len():
    out = onnx_greedy_kv(_FakeEncoder(), _FakeKV(), "x", max_len=4096)
    # 2-token cycle caught by the window guard well before max_len
    assert len(out) < 200
    assert out[:4] == [10, 11, 10, 11]


def test_normal_generation_unaffected():
    class _StopKV(_FakeKV):
        def run(self, _, feeds):
            import numpy as np

            logits = np.full((1, 1, 260), -1e9)
            if self.step >= 5:
                logits[0, -1, 1] = 1e9  # EOS_ID
            else:
                logits[0, -1, 20 + self.step] = 1e9
            self.step += 1
            return [logits, np.zeros((1,)), np.zeros((1,))]

    out = onnx_greedy_kv(_FakeEncoder(), _StopKV(), "x", max_len=256)
    assert out == [20, 21, 22, 23, 24]


def test_varying_separator_loop_is_cut():
    """Phrase + rotating punctuation never repeats a verbatim token
    window — the live int8 failure mode. The decoded-text guard cuts it."""

    class _RotateKV(_FakeKV):
        seps = ['"', " ", "\n", ":"]

        def run(self, _, feeds):
            import numpy as np

            logits = np.full((1, 1, 260), -1e9)
            if self.step == 0:
                logits[0, -1, 100] = 1e9  # phrase token
            else:
                mod = self.step % 4
                if mod == 0:
                    logits[0, -1, 100] = 1e9  # phrase again
                else:
                    # rotating separator tokens 200..203
                    logits[0, -1, 200 + ((self.step // 4) % 4)] = 1e9
            self.step += 1
            return [logits, np.zeros((1,)), np.zeros((1,))]

    out = onnx_greedy_kv(_FakeEncoder(), _RotateKV(), "x", max_len=4096)
    assert len(out) < 300
