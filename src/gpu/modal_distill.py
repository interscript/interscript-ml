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
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(str(REPO_ROOT), "/root/ml-models", copy=True)
    .workdir("/root/ml-models")
)

CHECKPOINTS = modal.Volume.from_name("rababa-checkpoints")
DATASETS = modal.Volume.from_name("rababa-datasets")
SECRYST_CHECKPOINTS = modal.Volume.from_name("secryst-checkpoints")
SECRYST_DATASETS = modal.Volume.from_name("secryst-datasets")
PERSIAN_CHECKPOINTS = modal.Volume.from_name("persian-g2p-checkpoints")

SPECS: dict[str, dict[str, str]] = {
    "tha-g2p-small": {
        # secryst's saved umt5 artifacts are unusable (5.15 dropped the
        # untied lm_head) and volume epitran data is tone-less (see
        # modal_teacher_thai.py notes) — teacher is the B-K hub base
        # directly: 6.37% PER published, verified loading + exact-match
        # generations under transformers 5.14.1 (probes 2026-08-18)
        "teacher": "B-K/umt5-thai-g2p-v2-0.5k",
        "teacher_is_hub": "true",
        "teacher_volume": "secryst",
        # run-004: ByT5-base student — the small student hit a 12.06%
        # generalization ceiling (+7.6pp over teacher, rejected); base
        # has 4x capacity at a ~1.2GB artifact, still client-tier
        "student_init": "google/byt5-base",
        "train": "thai-ipa-expanded/train.jsonl",
        "train_extra": ["thai-ipa/train.jsonl", "thai-ipa/augmented_epitran.jsonl"],
        "val": "thai-ipa-expanded/val.jsonl",
        "test": "thai-ipa-expanded/test.jsonl",
        "eval_test": "thai-ipa/test.jsonl",
        "out": "secryst_thai_g2p_distill_small/run-004",
        "mode": "sequence",  # cross-tokenizer: teacher generates, student trains CE
        "note": "umt5 (sentencepiece) teacher -> ByT5-small byte student; +5pp PER gate",
    },
    "fas-g2p-small": {
        "teacher": "persian_g2p/run-001/best",
        "teacher_volume": "persian",
        "student_init": "google/byt5-small",
        "train": "persian_g2p/train.jsonl",
        "val": "persian_g2p/val.jsonl",
        "test": "persian_g2p/test.jsonl",
        "out": "interscript_fas_g2p_distill_small/run-001",
        "note": "ByT5-small teacher (already byte-level) -> ByT5-small student; CER gate",
    },
    "heb-diac-small": {
        "teacher": "rababa_hebrew_byt5_s43/run-001/best",
        "student_init": "google/byt5-small",
        "train": "hebrew-v4/train.jsonl",
        "val": "hebrew-v4/val.jsonl",
        "out": "rababa_hebrew_distill_small/run-001",
    },
}

app = modal.App("interscript-ml-distill", image=IMAGE)


def decode_joined(tok, ids) -> str:
    """Correct decode for umt5 teachers: 5.x batch_decode inserts spurious
    spaces between sentencepiece pieces; pieces must join directly (the
    targets are unspaced IPA strings)."""
    skip = {tok.pad_token, tok.eos_token, tok.bos_token}
    return "".join(p for p in tok.convert_ids_to_tokens(ids) if p not in skip)


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
    teacher_vol = spec.get("teacher_volume", "rababa")
    vol_map = {
        "rababa": "/checkpoints",
        "secryst": "/secryst-checkpoints",
        "persian": "/persian-checkpoints",
    }
    data_vol = "/secryst-datasets" if teacher_vol == "secryst" else "/datasets"
    teacher_path = (spec["teacher"] if spec.get("teacher_is_hub")
                    else str(Path(vol_map[teacher_vol]) / spec["teacher"]))
    student_path = Path(vol_map[teacher_vol]) / spec["out"] / "best"
    test_path = Path(data_vol) / spec.get("eval_test", spec["test"])

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
    timeout=5 * 3600,
    volumes={
        "/datasets": DATASETS,
        "/checkpoints": CHECKPOINTS,
        "/secryst-checkpoints": SECRYST_CHECKPOINTS,
        "/secryst-datasets": SECRYST_DATASETS,
        "/persian-checkpoints": PERSIAN_CHECKPOINTS,
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

    vol_map = {
        "rababa": "/checkpoints",
        "secryst": "/secryst-checkpoints",
        "persian": "/persian-checkpoints",
    }
    teacher_root = vol_map[teacher_vol]
    teacher_path = (spec["teacher"] if spec.get("teacher_is_hub")
                    else str(Path(teacher_root) / spec["teacher"]))

    data_vol = "/secryst-datasets" if teacher_vol == "secryst" else "/datasets"
    train_path = Path(data_vol) / spec["train"]

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
    student = AutoModelForSeq2SeqLM.from_pretrained(spec["student_init"])
    student.train()

    class Pairs(Dataset):
        def __init__(self, paths: Path | list[Path], max_len: int = 384):
            if isinstance(paths, Path):
                paths = [paths]
            self.rows = []
            seen = set()
            for path in paths:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    s = (row.get("src") or "").strip()
                    if s and s not in seen and len(s.encode()) <= max_len:
                        seen.add(s)
                        self.rows.append((s, (row.get("tgt") or "").strip()))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        # byte-level tokens: a 2,000-char Wikipedia sentence is 2,000
        # tokens — without truncation a single long pair OOMs the A10G
        src = student_tok(
            [s for s, _ in batch], padding=True, truncation=True,
            max_length=384, return_tensors="pt",
        )
        labels = student_tok(
            [t for _, t in batch], padding=True, truncation=True,
            max_length=384, return_tensors="pt",
        ).input_ids
        labels[labels == student_tok.pad_token_id] = -100
        return src.input_ids, src.attention_mask, labels

    train_files = [train_path] + [
        Path(data_vol) / p for p in spec.get("train_extra", [])
    ]
    train_ds = Pairs(train_files)
    print(f"[{spec_id}] train pairs: {len(train_ds)} from {len(train_files)} files", flush=True)

    # Step 1: teacher generates labels (beam-4) for the full corpus.
    # Resumable: evictions mid-labeling are routine on long jobs —
    # already-labeled srcs are skipped, the rest are appended.
    out_root = Path(teacher_root) / spec["out"]
    out_root.mkdir(parents=True, exist_ok=True)
    teacher_labels_path = out_root / "teacher_labels.jsonl"

    done: set[str] = set()
    if teacher_labels_path.exists():
        for line in teacher_labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["src"])
                except (json.JSONDecodeError, KeyError):
                    continue  # torn last line from an eviction
        print(f"[{spec_id}] resuming labels: {len(done)} already done", flush=True)

    todo = [(s, t) for s, t in train_ds.rows if s not in done]
    if todo:
        print(f"[{spec_id}] labeling {len(todo)} remaining...", flush=True)

        def label_batch(batch, max_len: int = 384):
            # lone-src OOM fallback truncates once, then skips: never
            # recurse on the same shape (torch 2.x renames the OOM
            # exception class, so match by message)
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
                        **enc, max_new_tokens=max_len, num_beams=4
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

        # deterministic token-budget batching: sort by length so long
        # srcs land in small batches — no OOM roulette
        todo.sort(key=lambda p: len(p[0].encode()))
        budget = 16 * 200
        batches: list[list[tuple[str, str]]] = []
        cur: list[tuple[str, str]] = []
        cur_max = 0
        for pair in todo:
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

        labeled = 0
        with teacher_labels_path.open("a", encoding="utf-8") as fh:
            for batch in batches:
                preds = label_batch(batch)
                for (src, _), pred in zip(batch, preds, strict=True):
                    if pred is not None:
                        fh.write(
                            json.dumps(
                                {"src": src, "teacher": pred.strip()},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                labeled += len(batch)
                if labeled <= 200 * 16 or labeled % 3200 < len(batch):
                    mem = torch.cuda.memory_allocated() / 2**30
                    print(
                        f"  labeled {labeled}/{len(todo)} (gpu {mem:.2f} GiB)",
                        flush=True,
                    )
                if labeled % 3200 < len(batch):
                    SECRYST_CHECKPOINTS.commit()
    else:
        print(f"[{spec_id}] teacher labels already complete", flush=True)

    # Step 2: student trains on teacher labels (teacher no longer
    # needed on GPU — free it before the training loop)
    teacher.to("cpu")
    torch.cuda.empty_cache()
    student.to("cuda")
    student.gradient_checkpointing_enable()
    teacher_labels = []
    for line in teacher_labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            # drop degenerate outputs (repetition junk hits the token cap)
            label = (row["teacher"] or "").strip()
            if label and len(label.encode()) <= 384:
                teacher_labels.append((row["src"], label))
    print(f"[{spec_id}] trainable label pairs: {len(teacher_labels)}", flush=True)

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
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps // 20, total_steps
    )

    save_every = 500
    step = 0
    ckpts = sorted(out_root.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
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


@app.local_entrypoint()
def eval_per(spec: str = "tha-g2p-small", limit: int = 0) -> None:
    print(evaluate_per.remote(spec, limit))
