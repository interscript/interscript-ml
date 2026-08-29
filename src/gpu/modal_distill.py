"""Modal app: logit distillation of byte-level seq2seq models (WO07).

Case study (work order): Hebrew ByT5-base s43 (teacher) -> ByT5-small
(student) on the same hebrew-v4 corpus the teacher was trained on.
Loss = alpha * T^2 * KL(teacher_soft || student_soft) + (1 - alpha) * CE.
The student initializes from google/byt5-small pretraining — a pretrained
backbone is essential (from-scratch ByT5-small plateaus ~13% PER in the
Thai ablations; mode collapse resists all fixes).

    modal run --detach src/gpu/modal_distill.py::main --spec heb-diac-small
    until modal run --detach src/gpu/modal_distill.py::main --spec heb-diac-small; do sleep 60; done

Checkpoints on rababa-checkpoints:/rababa_hebrew_distill_small/run-001
(periodic save + auto-resume from latest — server evictions are expected).
GPU is A10G; never competes with A100 training runs.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # torch 2.12.1 leaks GPU memory across generate() calls under
        # transformers 5.14.1 (labeling OOMs at ~21 GiB on a <1 GiB
        # model); the probe image with unpinned torch showed no leak
        "torch>=2.4,<3",
        "transformers==5.14.1",
        "pyyaml>=6.0",
        "numpy>=1.26",
        # arabic gate harness: Misraj evaluator + SadeedDiac-25 parquet
        "pyarabic",
        "prettytable",
        "pandas",
        "pyarrow",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(str(REPO_ROOT), "/root/interscript-ml", copy=True)
    .add_local_file(
        "/Users/mulgogi/src/interscript/rababa/sadeed_evaluator.py",
        "/opt/rababa/sadeed_evaluator.py",
        copy=True,
    )
    .add_local_dir(
        "/Users/mulgogi/src/interscript/rababa/data/sadeed-diac-25",
        "/opt/rababa/data/sadeed-diac-25",
        copy=True,
    )
    .workdir("/root/interscript-ml")
)

CHECKPOINTS = modal.Volume.from_name("rababa-checkpoints")



VOLUME_MOUNTS = {
    "rababa": "/checkpoints",
    "secryst": "/secryst-checkpoints",
    "persian": "/persian-checkpoints",
}
DATA_MOUNTS = {
    "secryst": "/secryst-datasets",
    "persian": "/persian-datasets",
}


def resolve_spec(spec: dict) -> dict:
    """Volume-relative paths for a spec — the single owner of "which
    volume does this teacher/student/dataset live on" (previously four
    pasted vol_map blocks)."""
    teacher_vol = spec.get("teacher_volume", "rababa")
    data_root = spec.get("data_volume", DATA_MOUNTS.get(teacher_vol, "/datasets"))
    teacher = (
        spec["teacher"] if spec.get("teacher_is_hub")
        else str(Path(VOLUME_MOUNTS[teacher_vol]) / spec["teacher"])
    )
    out_root = str(Path(VOLUME_MOUNTS[spec.get("out_volume", teacher_vol)]) / spec["out"])
    return {
        "teacher_vol": teacher_vol,
        "data_root": data_root,
        "teacher": teacher,
        "out_root": out_root,
        "best": str(Path(out_root) / "best"),
    }


def svd_stitch_state(wide: dict, narrow: dict) -> dict:
    """Closed-form width bridge: project pretrained wide weights into a
    narrow state dict, preserving the top singular subspace per tensor
    (microkimi protocol). 2-D: W' = U_o^T W V_i over leading singular
    directions; 1-D: leading slice; N-D (head-count tensors):
    leading-dimension slices."""
    import torch

    out = {}
    for name, tgt in narrow.items():
        src = wide.get(name)
        if src is None or tuple(src.shape) == tuple(tgt.shape):
            out[name] = (src if src is not None else tgt).clone()
            continue
        if src.dim() == 2:
            o, i = src.shape
            o2, i2 = tgt.shape
            w = src.float()
            u, _, vh = torch.linalg.svd(w, full_matrices=False)
            if o2 < o:
                w = u[:, :o2].T @ w
            if i2 < i:
                w = w @ vh[:i2, :].T
            if o2 > o or i2 > i:
                padded = torch.zeros((o2, i2), dtype=w.dtype)
                padded[: min(o, o2), : min(i, i2)] = w[: min(o, o2), : min(i, i2)]
                w = padded
            out[name] = w.to(tgt.dtype)
        elif src.dim() == 1:
            out[name] = src[: tgt.shape[0]].clone().to(tgt.dtype)
        else:
            out[name] = src[tuple(slice(0, s) for s in tgt.shape)].clone().to(tgt.dtype)
    return out


def _maybe_stitch(spec_id: str, spec: dict, student) -> None:
    """When a custom-width student also names a pretrained init, bridge
    the pretrained weights down instead of random init (the capacity
    law says init is the variable that matters — ara-diac-tiny run-005)."""
    if not spec.get("student_init"):
        return
    from transformers import AutoModelForSeq2SeqLM

    pretrained = AutoModelForSeq2SeqLM.from_pretrained(spec["student_init"])
    print(f"[{spec_id}] svd-stitching from {spec['student_init']}", flush=True)
    student.load_state_dict(svd_stitch_state(pretrained.state_dict(), student.state_dict()))
    del pretrained


def _ensure_src_path() -> None:
    # Modal copies the entry file to /root/<name>.py while the repo image
    # sits at /root/interscript-ml — cover both layouts before importing
    # sibling modules (gpu.pkm, gpu.muon).
    import sys
    from pathlib import Path

    for cand in (
        Path(__file__).resolve().parent.parent,
        Path.cwd() / "src",
        Path("/root/interscript-ml/src"),
    ):
        if (cand / "gpu" / "pkm.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return
    raise RuntimeError("src/gpu/pkm.py not found on any known layout")
DATASETS = modal.Volume.from_name("rababa-datasets")
SECRYST_CHECKPOINTS = modal.Volume.from_name("secryst-checkpoints")
SECRYST_DATASETS = modal.Volume.from_name("secryst-datasets")
PERSIAN_CHECKPOINTS = modal.Volume.from_name("persian-g2p-checkpoints")
PERSIAN_DATASETS = modal.Volume.from_name("persian-g2p-datasets")

def _load_specs() -> dict:
    """SPECS as data (distill_specs.yaml) — the entry file is copied to
    /root/<name>.py on Modal while the repo image sits at
    /root/interscript-ml, so try both layouts."""
    import yaml

    for cand in (
        Path(__file__).resolve().parent / "distill_specs.yaml",
        Path.cwd() / "src/gpu/distill_specs.yaml",
        Path("/root/interscript-ml/src/gpu/distill_specs.yaml"),
    ):
        if cand.exists():
            return yaml.safe_load(cand.read_text(encoding="utf-8"))
    raise RuntimeError("distill_specs.yaml not found on any known layout")


SPECS: dict[str, dict[str, str]] = _load_specs()

app = modal.App("interscript-ml-distill", image=IMAGE)


def decode_joined(tok, ids) -> str:
    """Decode teacher generations correctly per tokenizer family.

    umt5/sentencepiece: 5.x batch_decode inserts spurious spaces between
    pieces; joining convert_ids_to_tokens is correct.

    byt5/byte-level: convert_ids_to_tokens returns each byte token as a
    RAW CHARACTER (latin-1 view) — joining them produces mojibake. This
    poisoned every Arabic label generated under 5.14 (both students
    trained on double-encoded targets and scored the identical bare-text
    DER). batch_decode is byte-exact here, so branch on it: if its
    output round-trips the same token ids it is authoritative.
    """
    skip = {tok.pad_token, tok.eos_token, tok.bos_token}
    joined = "".join(p for p in tok.convert_ids_to_tokens(ids) if p not in skip)
    if joined and all(ord(c) < 256 for c in joined):
        # byte-level vocab decoded to raw chars — use the byte-exact path
        return tok.batch_decode([ids], skip_special_tokens=True)[0]
    return joined


@app.function(
    gpu="A10G",
    cpu=8,
    memory=32 * 1024,
    timeout=6 * 3600,
    volumes={"/datasets": DATASETS, "/checkpoints": CHECKPOINTS},
)
def distill(spec_id: str, epochs: int = 3, alpha: float = 0.5, temperature: float = 2.0) -> dict:
    import json
    import math

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

    spec = SPECS[spec_id]
    device = "cuda"
    out_root = Path("/checkpoints") / spec["out"]
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    teacher = AutoModelForSeq2SeqLM.from_pretrained(
        Path("/checkpoints") / spec["teacher"], attn_implementation="eager"
    ).to(device, dtype=torch.float16).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if "student_config" in spec:
        # tiny tier: no pretrained backbone at this width — random init
        # from an explicit config (dense teacher-label supervision, see
        # the spec note on collapse risk)
        from transformers import T5Config, T5ForConditionalGeneration

        cfg = T5Config(**spec["student_config"])
        student = T5ForConditionalGeneration(cfg).to(device)
    else:
        student = AutoModelForSeq2SeqLM.from_pretrained(spec["student_init"]).to(device)
    student.train()

    class Pairs(Dataset):
        def __init__(self, path: Path, max_len: int = 384):
            self.rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s, t = (row.get("src") or "").strip(), (row.get("tgt") or "").strip()
                if s and t and len(s.encode()) <= max_len and len(t.encode()) <= max_len:
                    self.rows.append((s, t))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            s, t = self.rows[i]
            return s, t

    def collate(batch):
        src = tokenizer([s for s, _ in batch], padding=True, return_tensors="pt")
        labels = tokenizer([t for _, t in batch], padding=True, return_tensors="pt").input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        return src.input_ids, src.attention_mask, labels

    train_loader = DataLoader(
        Pairs(Path("/datasets") / spec["train"]),
        batch_size=8,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        drop_last=True,
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs

    # resume from the newest periodic checkpoint if present
    start_step = 0
    ckpts = sorted(out_root.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        state = torch.load(ckpts[-1] / "student.pt", map_location=device, weights_only=True)
        student.load_state_dict(state)
        opt_state = torch.load(ckpts[-1] / "optim.pt", map_location=device, weights_only=True)
        start_step = int(ckpts[-1].name.split("-")[1])
        print(f"[resume] from {ckpts[-1].name}", flush=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    if ckpts:
        optimizer.load_state_dict(opt_state)
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps // 20, total_steps)
    for _ in range(start_step):
        scheduler.step()

    save_every = 500
    log_every = 50
    step = start_step
    best_val = math.inf

    def val_loss() -> float:
        student.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for _, (ids, am, labels) in enumerate(
                DataLoader(
                    Pairs(Path("/datasets") / spec["val"]),
                    batch_size=8,
                    collate_fn=collate,
                )
            ):
                ids, am, labels = ids.to(device), am.to(device), labels.to(device)
                loss = student(
                    input_ids=ids, attention_mask=am, labels=labels
                ).loss
                total += float(loss)
                n += 1
                if n >= 50:  # sample the val split, it is only for model selection
                    break
        student.train()
        return total / max(n, 1)

    while step < total_steps:
        for ids, am, labels in train_loader:
            if step >= total_steps:
                break
            ids, am, labels = ids.to(device), am.to(device), labels.to(device)
            with torch.no_grad():
                t_logits = teacher(input_ids=ids, attention_mask=am, labels=labels).logits
            s_out = student(input_ids=ids, attention_mask=am, labels=labels)
            mask = labels != -100
            kd = F.kl_div(
                F.log_softmax(s_out.logits[mask] / temperature, dim=-1),
                F.softmax(t_logits.float()[mask] / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            loss = alpha * kd + (1 - alpha) * s_out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % log_every == 0:
                print(
                    f"[step {step}/{total_steps}] loss={float(loss):.4f} "
                    f"ce={float(s_out.loss):.4f} kd={float(kd):.4f}",
                    flush=True,
                )
            if step % save_every == 0:
                ck = out_root / f"step-{step}"
                ck.mkdir(exist_ok=True)
                torch.save(student.state_dict(), ck / "student.pt")
                torch.save(optimizer.state_dict(), ck / "optim.pt")
                CHECKPOINTS.commit()

    vl = val_loss()
    if vl < best_val:
        best = out_root / "best"
        best.mkdir(exist_ok=True)
        student.save_pretrained(str(best))
        tokenizer.save_pretrained(str(best))
        with (best / "val_loss.txt").open("w") as fh:
            fh.write(f"{vl}\n")
        CHECKPOINTS.commit()
    return {"spec": spec_id, "steps": step, "val_loss": vl}


NIKUD = None


def _nikud_only(text: str) -> list[str]:
    import re

    return re.findall(r"[\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7]", text)


def _edit_distance(a, b) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        curr = [i]
        for j, bj in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ai != bj)))
        prev = curr
    return prev[-1]


@app.function(
    gpu="A10G",
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 3600,
    volumes={"/datasets": DATASETS, "/checkpoints": CHECKPOINTS},
)
def evaluate(spec_id: str = "heb-diac-small", limit: int = 0) -> dict:
    """Greedy DER/CER of teacher and student on the same test pairs, one
    harness — the before/after for RESULTS.md."""
    import json
    from pathlib import Path

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    spec = SPECS[spec_id]
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    device = "cuda"
    teacher = AutoModelForSeq2SeqLM.from_pretrained(
        Path("/checkpoints") / spec["teacher"], attn_implementation="eager"
    ).to(device, dtype=torch.float16).eval()
    student = AutoModelForSeq2SeqLM.from_pretrained(
        Path("/checkpoints") / spec["out"] / "best"
    ).to(device, dtype=torch.float16).eval()

    pairs = []
    for line in (Path("/datasets") / "nakdimon" / "test-imf.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            row = json.loads(line)
            pairs.append((row["src"], row["tgt"]))
    if limit:
        pairs = pairs[:limit]

    def greedy(model, text: str, max_len: int = 256) -> str:
        ids = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_len, num_beams=1)
        return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    def metrics(model) -> dict:
        der_sum = cer_sum = n = 0.0
        for src, tgt in pairs:
            pred = greedy(model, src)
            gold_n, pred_n = _nikud_only(tgt), _nikud_only(pred)
            der_sum += _edit_distance(pred_n, gold_n) / max(1, len(gold_n))
            cer_sum += _edit_distance(list(pred), list(tgt)) / max(1, len(tgt))
            n += 1
        return {"der": round(100 * der_sum / n, 2), "cer": round(100 * cer_sum / n, 2), "n": int(n)}

    return {"teacher": metrics(teacher), "student": metrics(student)}




@app.function(
    gpu="A10G",
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 3600,
    volumes={
        "/datasets": DATASETS,
        "/checkpoints": CHECKPOINTS,
        "/secryst-checkpoints": SECRYST_CHECKPOINTS,
        "/secryst-datasets": SECRYST_DATASETS,
        "/persian-checkpoints": PERSIAN_CHECKPOINTS,
    },
)
def evaluate_per(spec_id: str, limit: int = 0) -> dict:
    """PER of teacher and student on the same g2p test split, replicating
    the source-model harness exactly (train_thai_combined.py::evaluate):
    held-out test file, beam-4 decode, corpus-level PER =
    total_ed / total_gold over whitespace tokens. The student gate is
    teacher_per + 5pp (docs/DISTILL-SOURCE-PROMPT.md)."""
    import json
    from pathlib import Path

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    spec = SPECS[spec_id]
    paths = resolve_spec(spec)
    data_vol = paths["data_root"]
    teacher_path = paths["teacher"]
    student_path = Path(paths["best"])
    test_rel = spec.get("eval_test") or spec.get("test")
    if not test_rel:
        raise RuntimeError(f"{spec_id}: no test path")
    test_path = Path(data_vol) / test_rel

    teacher_tok = AutoTokenizer.from_pretrained(teacher_path)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher_path).to("cuda").eval()
    student_tok = AutoTokenizer.from_pretrained("google/byt5-small")
    student = (
        AutoModelForSeq2SeqLM.from_pretrained(str(student_path)).to("cuda").eval()
    )

    pairs = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pairs.append((row["src"], row["tgt"]))
    if limit:
        pairs = pairs[:limit]
    print(f"[{spec_id}] eval pairs: {len(pairs)} from {test_path}", flush=True)

    def beam(tok, model, batch: list[str], max_len: int = 256,
             joined: bool = False) -> list[str]:
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_len, num_beams=4)
        if joined:
            return [decode_joined(tok, o) for o in out]
        return tok.batch_decode(out, skip_special_tokens=True)

    def per(model, tok, debug_name: str, joined: bool = False) -> dict:
        total_ed = total_gold = exact = n = 0
        for start in range(0, len(pairs), 32):
            batch = pairs[start : start + 32]
            preds = beam(tok, model, [src for src, _ in batch], joined=joined)
            for (_, gold), pred in zip(batch, preds, strict=True):
                e = _edit_distance(pred.strip().split(), gold.strip().split())
                total_ed += e
                total_gold += max(1, len(gold.split()))
                exact += e == 0
                n += 1
            if start == 0:
                for (src, gold), pred in zip(batch[:3], preds[:3], strict=True):
                    print(
                        f"[{debug_name}] src={src!r}\n  gold={gold!r}\n  pred={pred!r}",
                        flush=True,
                    )
        return {
            "per": round(100 * total_ed / max(1, total_gold), 2),
            "exact_match": round(100 * exact / max(1, n), 2),
            "n": n,
        }

    # cross-tokenizer (sequence-mode) teachers are sentencepiece umt5s —
    # they need the joined-piece decode; ByT5 byte students do not
    teacher_joined = spec.get("mode") == "sequence"
    result = {
        "teacher": per(teacher, teacher_tok, "teacher", joined=teacher_joined),
        "student": per(student, student_tok, "student"),
    }
    result["gate_delta"] = round(result["student"]["per"] - result["teacher"]["per"], 2)
    result["gate_pass"] = result["gate_delta"] <= 5.0
    return result


@app.function(
    gpu="A10G",
    cpu=8,
    memory=32 * 1024,
    timeout=12 * 3600,
    volumes={
        "/datasets": DATASETS,
        "/checkpoints": CHECKPOINTS,
        "/secryst-checkpoints": SECRYST_CHECKPOINTS,
        "/secryst-datasets": SECRYST_DATASETS,
        "/persian-checkpoints": PERSIAN_CHECKPOINTS,
        "/persian-datasets": PERSIAN_DATASETS,
    },
)
def distill_sequence(spec_id: str, epochs: int = 3) -> dict:
    """Cross-tokenizer sequence-level distillation: teacher generates
    labels with ITS tokenizer, student trains CE on those labels with
    the byte tokenizer. The only sound approach when teacher and student
    occupy different vocab spaces (umt5 sentencepiece vs ByT5 bytes)."""
    import json
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    spec = SPECS[spec_id]
    teacher_vol = spec.get("teacher_volume", "rababa")

    paths = resolve_spec(spec)
    teacher_path = paths["teacher"]
    train_path = Path(paths["data_root"]) / spec["train"]

    # Teacher: use its OWN tokenizer (sentencepiece for umt5)
    teacher_tok = AutoTokenizer.from_pretrained(str(teacher_path))
    teacher = (
        AutoModelForSeq2SeqLM.from_pretrained(str(teacher_path))
        .to("cuda", dtype=torch.float16)
        .eval()
    )
    for p in teacher.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(
        f"[{spec_id}] teacher loaded: {n_params:.0f}M params, "
        f"gpu {torch.cuda.memory_allocated() / 2**30:.2f} GiB",
        flush=True,
    )

    # Student: byte-level ByT5. Kept on CPU during labeling — only the
    # teacher needs the GPU there; eviction-prone A10G headroom matters.
    student_tok = AutoTokenizer.from_pretrained("google/byt5-small")
    if spec.get("student_config"):
        from transformers import T5Config, T5ForConditionalGeneration

        cfg = spec["student_config"]
        config = T5Config(
            vocab_size=259,
            d_model=cfg.get("d_model", 384),
            d_ff=cfg.get("d_ff", 1536),
            d_kv=cfg.get("d_model", 384) // cfg.get("num_heads", 6),
            num_layers=cfg.get("enc_layers", 8),
            num_decoder_layers=cfg.get("dec_layers", 8),
            num_heads=cfg.get("num_heads", 6),
            dropout_rate=0.1,
            feed_forward_proj=cfg.get("feed_forward_proj", "relu"),
            decoder_start_token_id=0,
            relative_attention_max_distance=128,
        )
        student = T5ForConditionalGeneration(config)
        _maybe_stitch(spec_id, spec, student)
        n_params = sum(q.numel() for q in student.parameters()) / 1e6
        print(f"[{spec_id}] tiny student: {n_params:.1f}M params", flush=True)
    else:
        student = AutoModelForSeq2SeqLM.from_pretrained(spec["student_init"])
    if spec.get("pkm"):
        _ensure_src_path()
        from gpu.pkm import inject_pkm

        inject_pkm(student, **spec["pkm"])
    student.train()

    class Pairs(Dataset):
        def __init__(self, files: list[tuple[Path, int]], max_len: int = 1450):
            # jsonl files carry {src, tgt} rows; .txt unit files are
            # single-column diacritized paragraph units (the r5 corpus):
            # src = diacritics stripped, tgt = the unit itself, capped at
            # max_len bytes, seeded shuffle then per-file limit
            import random
            import re

            self.rows = []
            seen = set()
            for path, limit in files:
                if path.suffix == ".jsonl":
                    rows = []
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        s = (row.get("src") or "").strip()
                        if s and s not in seen and len(s.encode()) <= 384:
                            seen.add(s)
                            rows.append((s, (row.get("tgt") or "").strip()))
                    if limit:
                        random.Random(42).shuffle(rows)
                        rows = rows[:limit]
                    self.rows.extend(rows)
                else:
                    diac = re.compile("[ً-ٰٟۖ-ۭ]")
                    units = [
                        u.strip()
                        for u in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        if u.strip()
                    ]
                    random.Random(42).shuffle(units)
                    for unit in units[:limit]:
                        if len(unit.encode()) > max_len:
                            continue
                        src = diac.sub("", unit).strip()
                        if src and src not in seen:
                            seen.add(src)
                            self.rows.append((src, unit))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    train_cap = int(spec.get("max_len", 384))

    def collate(batch):
        # byte-level tokens: a 2,000-char Wikipedia sentence is 2,000
        # tokens — without truncation a single long pair OOMs the A10G
        src = student_tok(
            [s for s, _ in batch], padding=True, truncation=True,
            max_length=train_cap, return_tensors="pt",
        )
        labels = student_tok(
            [t for _, t in batch], padding=True, truncation=True,
            max_length=train_cap, return_tensors="pt",
        ).input_ids
        labels[labels == student_tok.pad_token_id] = -100
        return src.input_ids, src.attention_mask, labels

    unit_limits = [int(x) for x in spec.get("unit_limits", [0])]
    train_files = [(train_path, unit_limits[0] if unit_limits else 0)]
    for i, p in enumerate(spec.get("train_extra", [])):
        lim = unit_limits[i + 1] if i + 1 < len(unit_limits) else 0
        train_files.append((Path(paths["data_root"]) / p, lim))
    train_ds = Pairs(train_files)
    print(f"[{spec_id}] train pairs: {len(train_ds)} from {len(train_files)} files", flush=True)
    label_beams = int(spec.get("label_beams", 4))

    # Step 1: teacher generates labels (beam-4) for the full corpus.
    # Resumable: evictions mid-labeling are routine on long jobs —
    # already-labeled srcs are skipped, the rest are appended.
    out_root = Path(paths["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    labels_file = spec.get("labels_file", "teacher_labels.jsonl")
    if labels_file.endswith(".b64"):
        # non-ASCII text is re-encoded somewhere in the modal
        # transfer layers (image COPY and volume put both mojibake'd a
        # UTF-8 jsonl to 3 parseable srcs); ship labels gzip+base64 and
        # decode to plain container-local bytes
        import base64
        import gzip

        teacher_labels_path = Path("/tmp/labels_decoded.jsonl")
        if not teacher_labels_path.exists():
            raw = gzip.decompress(
                base64.b64decode(Path(labels_file).read_text(encoding="ascii"))
            )
            teacher_labels_path.write_bytes(raw)
            print(f"[{spec_id}] decoded {len(raw)} label bytes", flush=True)
    else:
        teacher_labels_path = (
            Path(labels_file)
            if labels_file.startswith("/")
            else out_root / labels_file
        )

    def read_label_srcs() -> set[str]:
        # volume replicas can serve a stale view of a large file; retry
        # and keep the best parse rather than relabeling from zero
        import time as _time

        best: set[str] = set()
        for _attempt in range(3):
            got: set[str] = set()
            for line in teacher_labels_path.read_text(
                encoding="utf-8", errors="ignore"
            ).split("\n"):
                if line.strip():
                    try:
                        got.add(json.loads(line)["src"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            if len(got) > len(best):
                best = got
            _time.sleep(20)
        return best

    done: set[str] = set()
    if spec.get("labels_complete") and teacher_labels_path.exists():
        print(f"[{spec_id}] labels trusted complete", flush=True)
        done = {s_ for s_, _ in train_ds.rows}
    elif teacher_labels_path.exists():
        done = read_label_srcs()
        print(f"[{spec_id}] resuming labels: {len(done)} already done", flush=True)

    todo = [(s, t) for s, t in train_ds.rows if s not in done]
    fresh_rows: list[tuple[str, str]] = []
    seq_max = int(spec.get("max_len", 384))

    def label_batch(batch, max_len: int = 0):
        # lone-src OOM fallback truncates once, then skips: never
        # recurse on the same shape (torch 2.x renames the OOM
        # exception class, so match by message)
        if not max_len:
            max_len = seq_max
        try:
            enc = teacher_tok(
                [s for s, _ in batch],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                out = teacher.generate(
                    # r5 contract: generation cap = 2x window bytes
                    # (diacritized output runs 1.4-1.6x input)
                    **enc, max_new_tokens=2 * max_len, num_beams=label_beams
                )
            return [decode_joined(teacher_tok, o) for o in out]
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            if len(batch) == 1:
                if max_len > 128:
                    return label_batch(batch, max_len=128)
                print(f"  [{spec_id}] skipping pathological src", flush=True)
                return [None]
            mid = len(batch) // 2
            return label_batch(batch[:mid], max_len) + label_batch(
                batch[mid:], max_len
            )

    def label_all(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        # deterministic token-budget batching: sort by length so long
        # srcs land in small batches — no OOM roulette
        pairs = sorted(pairs, key=lambda p: len(p[0].encode()))
        budget = 32 * max(200, seq_max)
        batches: list[list[tuple[str, str]]] = []
        cur: list[tuple[str, str]] = []
        cur_max = 0
        for pair in pairs:
            length = len(pair[0].encode())
            new_max = max(cur_max, length)
            if cur and (len(cur) + 1) * new_max > budget:
                batches.append(cur)
                cur, cur_max = [], 0
                new_max = length
            cur.append(pair)
            cur_max = new_max
        if cur:
            batches.append(cur)

        rows: list[tuple[str, str]] = []
        labeled = 0
        with teacher_labels_path.open("a", encoding="utf-8") as fh:
            for batch in batches:
                preds = label_batch(batch)
                for (src, _), pred in zip(batch, preds, strict=True):
                    if pred is not None:
                        text = pred.strip()
                        fh.write(
                            json.dumps(
                                {"src": src, "teacher": text},
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
                        rows.append((src, text))
                labeled += len(batch)
                if labeled <= 200 * 16 or labeled % 3200 < len(batch):
                    mem = torch.cuda.memory_allocated() / 2**30
                    print(
                        f"  labeled {labeled}/{len(pairs)} (gpu {mem:.2f} GiB)",
                        flush=True,
                    )
                if labeled % 3200 < len(batch):
                    SECRYST_CHECKPOINTS.commit()
        # the modulo can leave the tail uncommitted
        {
            "secryst": SECRYST_CHECKPOINTS,
            "rababa": CHECKPOINTS,
            "persian": PERSIAN_CHECKPOINTS,
        }.get(spec.get("out_volume", teacher_vol), SECRYST_CHECKPOINTS).commit()
        return rows

    if todo:
        print(f"[{spec_id}] labeling {len(todo)} remaining...", flush=True)
        fresh_rows = label_all(todo)
    else:
        print(f"[{spec_id}] teacher labels already complete", flush=True)

    # Step 2: student trains on teacher labels. The teacher stays on the
    # GPU until the labels are confirmed usable — regeneration (below)
    # still needs it.
    label_cap = 2 * int(spec.get("max_len", 384))
    teacher_labels = []
    seen_labels: set[str] = set()

    def accept_label(src: str, label: str) -> None:
        src, label = src.strip(), label.strip()
        if src and src not in seen_labels and label and len(label.encode()) <= label_cap:
            seen_labels.add(src)
            teacher_labels.append((src, label))

    if not fresh_rows:
        for line in teacher_labels_path.read_text(encoding="utf-8", errors="ignore").split("\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn line from a volume replication race
            accept_label(row.get("src") or "", row.get("teacher") or "")
        if len(teacher_labels) < 0.5 * len(train_ds.rows):
            # the file is unusable from this container: regenerate rather
            # than fail (relaunch-only loops forever on this path)
            print(
                f"[{spec_id}] labels unusable ({len(teacher_labels)} valid "
                f"pairs); regenerating all labels",
                flush=True,
            )
            teacher_labels = []
            seen_labels = set()
            fresh_rows = label_all(list(train_ds.rows))
    if fresh_rows:
        for src, label in fresh_rows:
            accept_label(src, label)
    print(f"[{spec_id}] trainable label pairs: {len(teacher_labels)}", flush=True)
    if len(teacher_labels) < 0.5 * len(train_ds.rows):
        raise RuntimeError(
            f"labels unusable even after regeneration: "
            f"{len(teacher_labels)} valid pairs for {len(train_ds.rows)} srcs"
        )

    teacher.to("cpu")
    torch.cuda.empty_cache()
    student.to("cuda")
    student.gradient_checkpointing_enable()

    class TeacherPairs(Dataset):
        def __len__(self):
            return len(teacher_labels)

        def __getitem__(self, i):
            return teacher_labels[i]

    train_loader = DataLoader(
        TeacherPairs(),
        batch_size=8,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        drop_last=True,
    )
    total_steps = len(train_loader) * epochs
    if spec.get("optimizer") == "muon":
        _ensure_src_path()
        from gpu.muon import Muon, split_parameters

        muon_params, adamw_params = split_parameters(student.named_parameters())
        optimizer = Muon(
            muon_params, lr=float(spec.get("muon_lr", 0.01)),
            momentum=0.95, weight_decay=0.01,
        )
        optimizer.add_adamw_group(adamw_params, lr=1e-4, weight_decay=0.0)
        print(
            f"[{spec_id}] muon: {len(muon_params)} matrix / "
            f"{len(adamw_params)} embedding-like params",
            flush=True,
        )
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps // 20, total_steps
    )

    save_every = 500
    step = 0
    import hashlib

    labels_digest = hashlib.sha256(
        teacher_labels_path.read_bytes()
    ).hexdigest()[:12] if teacher_labels_path.exists() else "none"

    def _usable(ck: Path) -> bool:
        marker = ck / "labels.sha"
        return marker.exists() and marker.read_text().strip() == labels_digest

    ckpts = sorted(
        (c for c in out_root.glob("step-*") if _usable(c)),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if ckpts:
        student.load_state_dict(
            torch.load(ckpts[-1] / "student.pt", map_location="cpu", weights_only=True)
        )
        optimizer.load_state_dict(
            torch.load(ckpts[-1] / "optim.pt", map_location="cpu", weights_only=True)
        )
        step = int(ckpts[-1].name.split("-")[1])
        for _ in range(step):
            scheduler.step()
        print(f"[{spec_id}] resume training from step-{step}", flush=True)

    for _ in range(epochs):
        for ids, am, labels in train_loader:
            if step >= total_steps:
                break
            ids, am, labels = ids.to("cuda"), am.to("cuda"), labels.to("cuda")
            loss = student(input_ids=ids, attention_mask=am, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 50 == 0:
                print(
                    f"[{spec_id} step {step}/{total_steps}] ce={float(loss):.4f}",
                    flush=True,
                )
            if step % save_every == 0:
                ck = out_root / f"step-{step}"
                ck.mkdir(exist_ok=True)
                (ck / "labels.sha").write_text(labels_digest)
                torch.save(student.state_dict(), ck / "student.pt")
                torch.save(optimizer.state_dict(), ck / "optim.pt")
                CHECKPOINTS.commit()
                SECRYST_CHECKPOINTS.commit()
                PERSIAN_CHECKPOINTS.commit()

    best = out_root / "best"
    best.mkdir(exist_ok=True)
    student.save_pretrained(str(best))
    student_tok.save_pretrained(str(best))
    CHECKPOINTS.commit()
    SECRYST_CHECKPOINTS.commit()
    PERSIAN_CHECKPOINTS.commit()
    return {"spec": spec_id, "steps": step}


@app.local_entrypoint()
def main(spec: str = "heb-diac-small", epochs: int = 3) -> None:
    mode = SPECS[spec].get("mode", "logit")
    fn = distill_sequence if mode == "sequence" else distill
    result = fn.remote(spec, epochs=epochs)
    print(result)


@app.local_entrypoint()
def eval_main(spec: str = "heb-diac-small", limit: int = 0) -> None:
    print(evaluate.remote(spec, limit))


@app.function(
    gpu="A10G",
    cpu=8,
    memory=32 * 1024,
    timeout=5 * 3600,
    volumes={
        "/datasets": DATASETS,
        "/checkpoints": CHECKPOINTS,
        "/secryst-checkpoints": SECRYST_CHECKPOINTS,
        "/secryst-datasets": SECRYST_DATASETS,
        "/persian-checkpoints": PERSIAN_CHECKPOINTS,
    },
)
def evaluate_der(spec_id: str, window: int = 1400, limit: int = 0) -> dict:
    """Windowed SadeedDiac-25 DER-CE of teacher vs student, replicating
    rababa eval_sadeed_windowed.py at the r5 window (1400B): strip
    diacritics, split at word boundaries, greedy decode with 2x window
    cap, stitch, project haraqat onto the input letters (zero-skip),
    DER-CE via the Misraj evaluator. Gate: teacher + 0.5pp
    (DISTILL-SOURCE-PROMPT: 3.18 target from the 2.68 teacher)."""
    from pathlib import Path

    import pyarrow.parquet as pq
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _ensure_src_path()
    from harness.sadeed import (
        strip_diacritics,
        windowed_paragraphs,
    )

    spec = SPECS[spec_id]
    paths = resolve_spec(spec)
    teacher_path = paths["teacher"]
    student_path = Path(paths["best"])

    tok = AutoTokenizer.from_pretrained("google/byt5-small")
    teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher_path).to("cuda").eval()
    if spec.get("pkm"):
        _ensure_src_path()
        from gpu.pkm import load_student_with_pkm

        student = load_student_with_pkm(student_path, spec["pkm"]).to("cuda").eval()
    else:
        student = AutoModelForSeq2SeqLM.from_pretrained(str(student_path)).to("cuda").eval()
    # custom students carry T5's default max_length=20; windowed inputs
    # run to 1400 bytes, clamping max_new_tokens to zero
    for m in (teacher, student):
        m.generation_config.max_length = 100_000

    table = pq.read_table("/opt/rababa/data/sadeed-diac-25/train.parquet")
    inputs = [strip_diacritics(t) for t in table.column("input").to_pylist()]
    gts = table.column("output").to_pylist()
    if limit:
        inputs, gts = inputs[:limit], gts[:limit]

    def der_ce(model) -> dict:
        paragraphs = windowed_paragraphs(model, tok, inputs, window=window)

        import sys

        sys.path.insert(0, "/opt/rababa")
        from sadeed_evaluator import ArabicDiacritizationEvaluator as E

        _, _, total_der, _, _ = E.caculate_errors_on_sentences(
            paragraphs, gts, gt_missing_diacritic_is_error=False
        )
        return {"der_ce": round(total_der, 4), "n": len(inputs)}  # evaluator already returns %

    result = {"teacher": der_ce(teacher), "student": der_ce(student)}
    result["gate_delta"] = round(result["student"]["der_ce"] - result["teacher"]["der_ce"], 4)
    result["gate_pass"] = result["gate_delta"] <= 0.5
    # durable verdict marker: the run dir is the provenance record (also
    # what r7-style _init_choice probes read)
    import json

    out_root = Path(paths["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "final_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    CHECKPOINTS.commit()
    return result


@app.local_entrypoint()
def eval_per(spec: str = "tha-g2p-small", limit: int = 0) -> None:
    print(evaluate_per.remote(spec, limit))


@app.function(
    gpu="A10G",
    cpu=8,
    memory=32 * 1024,
    timeout=12 * 3600,
    volumes={
        "/datasets": DATASETS,
        "/checkpoints": CHECKPOINTS,
        "/secryst-checkpoints": SECRYST_CHECKPOINTS,
        "/secryst-datasets": SECRYST_DATASETS,
        "/persian-checkpoints": PERSIAN_CHECKPOINTS,
        "/persian-datasets": PERSIAN_DATASETS,
    },
)
def distill_microkimi(spec_id: str, epochs: int = 3, calib_batches: int = 64,
                      ridge_lambda: float = 1e-2, hidden_weight: float = 1.0) -> dict:
    """Bridge distillation (microkimi recipe): teacher and student share
    the byte tokenizer, so activations align token-for-token. Calibration
    collects per-layer-pair Gram stats; closed-form ridge solve gives
    frozen projectors teacher_h (d_t) -> student_h (d_s); training = CE +
    hidden-MSE through the frozen bridges. Rescues the from-scratch
    collapse seen at 33M params on G2P."""
    import json
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        T5Config,
        T5ForConditionalGeneration,
        get_cosine_schedule_with_warmup,
    )

    spec = SPECS[spec_id]
    paths = resolve_spec(spec)
    out_root = Path(paths["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    teacher_path = paths["teacher"]

    student_tok = AutoTokenizer.from_pretrained("google/byt5-small")
    teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher_path).to("cuda").eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    if spec.get("student_config"):
        cfg = spec["student_config"]
        config = T5Config(
            vocab_size=259,
            d_model=cfg.get("d_model", 384),
            d_ff=cfg.get("d_ff", 1536),
            d_kv=cfg.get("d_model", 384) // cfg.get("num_heads", 6),
            num_layers=cfg.get("enc_layers", 8),
            num_decoder_layers=cfg.get("dec_layers", 8),
            num_heads=cfg.get("num_heads", 6),
            dropout_rate=0.1,
            feed_forward_proj=cfg.get("feed_forward_proj", "relu"),
            decoder_start_token_id=0,
            relative_attention_max_distance=128,
        )
        student = T5ForConditionalGeneration(config)
        _maybe_stitch(spec_id, spec, student)
    else:
        student = AutoModelForSeq2SeqLM.from_pretrained(spec["student_init"])
    student.to("cuda").train()
    n_s = sum(q.numel() for q in student.parameters()) / 1e6
    print(f"[{spec_id}] microkimi: student {n_s:.1f}M params", flush=True)

    labels_file = out_root / spec.get("labels_file", "teacher_labels.jsonl")
    if not (labels_file.exists() and spec.get("labels_complete")):
        raise RuntimeError("microkimi expects pre-generated trusted labels")
    teacher_labels = []
    seen: set[str] = set()
    for line in labels_file.read_text(encoding="utf-8", errors="ignore").split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        src = (row.get("src") or "").strip()
        label = (row.get("teacher") or "").strip()
        if src and src not in seen and label and len(label.encode()) <= 384:
            seen.add(src)
            teacher_labels.append((src, label))
    print(f"[{spec_id}] trainable label pairs: {len(teacher_labels)}", flush=True)

    cap = int(spec.get("max_len", 384))

    def collate(batch):
        src = student_tok([s for s, _ in batch], padding=True, truncation=True,
                          max_length=cap, return_tensors="pt")
        labels = student_tok([t for _, t in batch], padding=True, truncation=True,
                             max_length=cap, return_tensors="pt").input_ids
        labels[labels == student_tok.pad_token_id] = -100
        return src.input_ids, src.attention_mask, labels

    class TeacherPairs(Dataset):
        def __len__(self):
            return len(teacher_labels)

        def __getitem__(self, i):
            return teacher_labels[i]

    loader = DataLoader(TeacherPairs(), batch_size=8, shuffle=True,
                        collate_fn=collate, num_workers=2, drop_last=True)

    t_enc = teacher.config.num_layers
    t_dec = teacher.config.num_decoder_layers or teacher.config.num_layers
    s_enc = student.config.num_layers
    s_dec = student.config.num_decoder_layers or student.config.num_layers
    enc_pairs = [(j, round(j * (t_enc - 1) / max(1, s_enc - 1))) for j in range(s_enc)]
    dec_pairs = [(j, round(j * (t_dec - 1) / max(1, s_dec - 1))) for j in range(s_dec)]
    d_t = teacher.config.d_model
    d_s = student.config.d_model

    def fwd(model, ids, am, labels):
        return model(input_ids=ids, attention_mask=am, labels=labels,
                     output_hidden_states=True)

    stats = {}
    for kind, pairs in (("enc", enc_pairs), ("dec", dec_pairs)):
        for j, _ in pairs:
            stats[(kind, j)] = [torch.zeros(d_t, d_t, device="cuda"),
                                torch.zeros(d_t, d_s, device="cuda"),
                                torch.zeros(d_t, device="cuda"),
                                torch.zeros(d_s, device="cuda"),
                                0]
    calib = DataLoader(TeacherPairs(), batch_size=8, shuffle=True,
                       collate_fn=collate, num_workers=2, drop_last=True)
    n_cal = 0
    with torch.no_grad():
        for i, (ids, am, labels) in enumerate(calib):
            if i >= calib_batches:
                break
            ids, am, labels = ids.to("cuda"), am.to("cuda"), labels.to("cuda")
            t_out = fwd(teacher, ids, am, labels)
            s_out = fwd(student, ids, am, labels)
            for kind, pairs, t_hs, s_hs in (
                ("enc", enc_pairs, t_out.encoder_hidden_states, s_out.encoder_hidden_states),
                ("dec", dec_pairs, t_out.decoder_hidden_states, s_out.decoder_hidden_states),
            ):
                for j, t_idx in pairs:
                    h_t = t_hs[t_idx].reshape(-1, d_t)
                    h_s = s_hs[j].reshape(-1, d_s)
                    st = stats[(kind, j)]
                    st[0] += h_t.T @ h_t
                    st[1] += h_t.T @ h_s
                    st[2] += h_t.sum(0)
                    st[3] += h_s.sum(0)
                    st[4] += h_t.shape[0]
            n_cal += ids.shape[0]
    print(f"[{spec_id}] calibration over {n_cal} pairs", flush=True)

    bridges = {}
    for key, (gxx, gxd, mx, ms, n) in stats.items():
        mean_t = mx / n
        mean_s = ms / n
        a = gxx + ridge_lambda * torch.eye(d_t, device="cuda") * gxx.diagonal().mean()
        w = torch.linalg.solve(a, gxd)
        b = mean_s - mean_t @ w
        bridges[key] = (w.detach(), b.detach())
        print(f"[{spec_id}] bridge {key} solved", flush=True)

    total_steps = len(loader) * epochs
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps // 20, total_steps)

    start_step = 0
    ckpts = sorted(out_root.glob("mk-step-*"), key=lambda q: int(q.name.split("-")[2]))
    if ckpts:
        student.load_state_dict(torch.load(ckpts[-1] / "student.pt", map_location="cpu",
                                           weights_only=True))
        student.to("cuda")
        optimizer.load_state_dict(torch.load(ckpts[-1] / "optim.pt", map_location="cpu",
                                             weights_only=True))
        start_step = int(ckpts[-1].name.split("-")[2])
        for _ in range(start_step):
            scheduler.step()
        print(f"[{spec_id}] resume microkimi from step-{start_step}", flush=True)

    step = start_step
    for _ in range(epochs):
        for ids, am, labels in loader:
            if step >= total_steps:
                break
            ids, am, labels = ids.to("cuda"), am.to("cuda"), labels.to("cuda")
            s_out = fwd(student, ids, am, labels)
            with torch.no_grad():
                t_out = fwd(teacher, ids, am, labels)
            loss = s_out.loss
            mse_total = s_out.loss.new_zeros(())
            n_tok = 0
            mask_e = (am == 1).unsqueeze(-1).float()
            for j, t_idx in enc_pairs:
                w, b = bridges[("enc", j)]
                target = t_out.encoder_hidden_states[t_idx] @ w + b
                diff = ((s_out.encoder_hidden_states[j] - target) ** 2 * mask_e).sum()
                mse_total = mse_total + diff
                n_tok += mask_e.sum() * d_s
            mask_d = (labels != -100).float().unsqueeze(-1)
            for j, t_idx in dec_pairs:
                w, b = bridges[("dec", j)]
                target = t_out.decoder_hidden_states[t_idx] @ w + b
                diff = ((s_out.decoder_hidden_states[j] - target) ** 2 * mask_d).sum()
                mse_total = mse_total + diff
                n_tok += mask_d.sum() * d_s
            hidden_loss = mse_total / (n_tok + 1e-9)
            total = loss + hidden_weight * hidden_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"[{spec_id} mk-step {step}/{total_steps}] "
                      f"ce={float(loss):.4f} hidden={float(hidden_loss):.6f}", flush=True)
            if step % 500 == 0:
                ck = out_root / f"mk-step-{step}"
                ck.mkdir(exist_ok=True)
                torch.save(student.state_dict(), ck / "student.pt")
                torch.save(optimizer.state_dict(), ck / "optim.pt")
                CHECKPOINTS.commit()
                SECRYST_CHECKPOINTS.commit()
                PERSIAN_CHECKPOINTS.commit()

    best = out_root / "best"
    best.mkdir(exist_ok=True)
    student.save_pretrained(str(best))
    student_tok.save_pretrained(str(best))
    CHECKPOINTS.commit()
    SECRYST_CHECKPOINTS.commit()
    PERSIAN_CHECKPOINTS.commit()
    return {"spec": spec_id, "steps": step, "mode": "microkimi"}


@app.local_entrypoint()
def mk(spec: str = "tha-g2p-tiny-mk", epochs: int = 3) -> None:
    print(distill_microkimi.remote(spec, epochs=epochs))


@app.local_entrypoint()
def eval_der(spec: str = "ara-diac-small", limit: int = 0) -> None:
    print(evaluate_der.remote(spec, limit=limit))


@app.function(
    cpu=4,
    memory=16 * 1024,
    timeout=30 * 60,
    volumes={"/checkpoints": CHECKPOINTS, "/secryst-checkpoints": SECRYST_CHECKPOINTS},
)
def probe_pkm_gates(spec_id: str = "ara-diac-small-pkm") -> dict:
    """E2 engagement probe (r8's IPA-probe analogue): gate values and
    memory-table statistics from the latest step checkpoint. Gates moving
    off zero = the memory branch is being used, not bypassed."""
    from pathlib import Path

    import torch
    from transformers import AutoModelForSeq2SeqLM

    spec = SPECS[spec_id]
    out_root = Path("/checkpoints") / spec["out"]
    ckpts = sorted(
        out_root.glob("step-*"), key=lambda p: int(p.name.split("-")[1])
    )
    if not ckpts:
        raise RuntimeError(f"no step checkpoints under {out_root}")

    _ensure_src_path()
    from gpu.pkm import inject_pkm

    student = AutoModelForSeq2SeqLM.from_pretrained(spec["student_init"])
    inject_pkm(student, **spec["pkm"])
    sd = torch.load(ckpts[-1] / "student.pt", map_location="cpu", weights_only=True)
    student.load_state_dict(sd)

    report: dict = {"checkpoint": ckpts[-1].name, "gates": {}, "tables": {}}
    for i, block in enumerate(student.decoder.block):
        wrapped = block.layer[1]
        if not hasattr(wrapped, "memory"):
            continue
        report["gates"][f"dec-{i}"] = round(float(wrapped.gate), 4)
        v = wrapped.memory.values
        report["tables"][f"dec-{i}"] = {
            "rows": int(v.shape[0]),
            "row_norm_mean": round(float(v.norm(dim=-1).mean()), 4),
            "row_norm_p99": round(float(v.norm(dim=-1).quantile(0.99)), 4),
            "key_drift_q1": round(float(wrapped.memory.k1.abs().mean()), 4),
        }
    return report


@app.local_entrypoint()
def pkm_gates(spec: str = "ara-diac-small-pkm") -> None:
    print(probe_pkm_gates.remote(spec))


@app.function(
    cpu=1,
    memory=1024,
    timeout=24 * 3600,
    volumes={"/checkpoints": CHECKPOINTS, "/secryst-checkpoints": SECRYST_CHECKPOINTS},
)
def qwen_next_chain() -> dict:
    """Server-side orchestrator for the qwen-next experiments (E2/E3):
    workstation-independent, self-healing, idempotent — the replacement
    for local shell chains that die with the workstation or break when
    the repo's checked-out branch changes.

    State machine per arm, driven by durable volume markers only:
      best/config.json absent  -> watch step-* checkpoints; respawn
                                   training if no progress for 20 min
                                   (distill_sequence resumes from the
                                   latest checkpoint; a redundant spawn
                                   is benign — a finished run saves
                                   best again and exits)
      best present, final_eval.json absent -> evaluate_der (which now
                                   writes final_eval.json itself)
      final_eval.json present  -> arm done

    Audit trail: chain_log.jsonl in each run dir. If this function times
    out (24h) or is evicted, relaunching continues from the markers:

        modal run --detach src/gpu/modal_distill.py::qwen_chain
    """
    import time
    from pathlib import Path

    ARMS = [
        ("ara-diac-small-pkm", "rababa_arabic_distill_small/run-003-pkm"),
        ("ara-diac-small-pkm-muon", "rababa_arabic_distill_small/run-004-pkm-muon"),
        ("ara-diac-small-muon", "rababa_arabic_distill_small/run-005-muon"),
        ("ara-diac-small-2", "rababa_arabic_distill_small/run-006-r7-muon"),
    ]

    _ensure_src_path()
    from gpu.runstate import RunState

    status = {}
    for spec_id, run in ARMS:
        state = RunState(Path("/checkpoints") / run)
        while not state.training_done():
            CHECKPOINTS.reload()
            before = state.latest_step()
            state.log(f"watch step={before}", commit=CHECKPOINTS.commit)
            time.sleep(1200)
            CHECKPOINTS.reload()
            after = state.latest_step()
            if after == before and not state.training_done():
                state.log(f"stalled at step={after}; respawning {spec_id}",
                          commit=CHECKPOINTS.commit)
                distill_sequence.spawn(spec_id, epochs=3)
        state.log("training complete (best present)", commit=CHECKPOINTS.commit)
        if not state.eval_done():
            state.log("evaluating", commit=CHECKPOINTS.commit)
            evaluate_der.remote(spec_id=spec_id)
            state.log("eval done", commit=CHECKPOINTS.commit)
        status[run] = "complete"
    return status


@app.local_entrypoint()
def qwen_chain() -> None:
    handle = qwen_next_chain.spawn()
    print(
        f"spawned {handle.object_id}; durable markers: chain_log.jsonl, "
        f"final_eval.json in each run dir; relaunch is idempotent",
        flush=True,
    )
