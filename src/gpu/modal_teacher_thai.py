"""Recover the Thai umt5 teacher (WO07 prerequisite).

The secryst umt5 checkpoints on secryst-checkpoints were saved under
transformers 5.15.0, which dropped the trained (untied) umt5 lm_head —
every saved artifact degenerates at inference (verified 2026-08-18;
published PERs 2.32/3.24% were live-eval numbers, not reproducible from
the saved files). This script re-finetunes the same recipe under
transformers 5.14.1 (the version all working ByT5 exports used):

  base   B-K/umt5-thai-g2p-v2-0.5k (HF hub, loads correctly)
  data   thai-ipa-expanded/train.jsonl (the 60K Kaikki+epitran mix)
  eval   thai-ipa/test.jsonl, beam-4, joined-piece decode, corpus PER

    modal run --detach src/gpu/modal_teacher_thai.py::main
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1",
        "transformers==5.14.1",
        "pyyaml>=6.0",
        "numpy>=1.26",
    )
    .add_local_dir(str(REPO_ROOT), "/root/ml-models", copy=True)
    .workdir("/root/ml-models")
)

CKPTS = modal.Volume.from_name("secryst-checkpoints")
DATA = modal.Volume.from_name("secryst-datasets")

BASE = "B-K/umt5-thai-g2p-v2-0.5k"
OUT = "secryst_thai_ipa_teacher_recovery/run-003"

app = modal.App("tha-teacher-recovery", image=IMAGE)


def _decode_joined(tok, ids) -> str:
    pieces = tok.convert_ids_to_tokens(ids)
    skip = {tok.pad_token, tok.eos_token, tok.bos_token}
    return "".join(p for p in pieces if p not in skip)


@app.function(
    gpu="A10G",
    cpu=8,
    memory=32 * 1024,
    timeout=6 * 3600,
    volumes={"/ckpts": CKPTS, "/datasets": DATA},
)
def train(epochs: int = 2, lr: float = 3e-5, batch: int = 16,
          stage: int = 1, warmup: int = 50, seed: int = 42, tag: str = "") -> dict:
    """Stage 1: Kaikki-only (9.7K, 10 ep, lr 3e-5 — the curriculum
    phase-1 recipe). Stage 2 is intentionally NOT used (the volume's
    epitran file is tone-less). `tag` variants allow seed selection."""
    import json
    import random

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    torch.manual_seed(seed)
    random.seed(seed)

    device = "cuda"
    out_root = Path("/ckpts") / OUT / f"stage{stage}{tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    if stage == 1:
        init = BASE
        train_paths = [Path("/datasets/thai-ipa/train.jsonl")]
        epochs = epochs if epochs != 2 else 10
    else:
        init = str(Path("/ckpts") / OUT / "stage1" / "best")
        train_paths = [
            Path("/datasets/thai-ipa/train.jsonl"),
            Path("/datasets/thai-ipa/augmented_epitran.jsonl"),
        ]

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(init).to(device)
    model.train()

    class Pairs(Dataset):
        def __init__(self, paths: list[Path], max_len: int = 384):
            self.rows = []
            for path in paths:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    s, t = (row.get("src") or "").strip(), (row.get("tgt") or "").strip()
                    if s and t:
                        self.rows.append((s, t))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        src = tok([s for s, _ in batch], padding=True, truncation=True,
                  max_length=384, return_tensors="pt")
        labels = tok([t for _, t in batch], padding=True, truncation=True,
                     max_length=384, return_tensors="pt").input_ids
        labels[labels == tok.pad_token_id] = -100
        return src.input_ids, src.attention_mask, labels

    # secryst's exact combined recipe (train_thai_combined.py):
    # Kaikki 9.7K + epitran-augmented 50K (stage 2); Kaikki only (stage 1)
    loader = DataLoader(Pairs(train_paths), batch_size=batch, shuffle=True,
                        collate_fn=collate, num_workers=2, drop_last=True)
    total_steps = len(loader) * epochs
    print(f"stage={stage} init={init} pairs={len(loader.dataset)} steps={total_steps}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    start_step = 0
    ckpts = sorted(out_root.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        model.load_state_dict(torch.load(ckpts[-1] / "model.pt", map_location=device,
                                         weights_only=True))
        optimizer.load_state_dict(torch.load(ckpts[-1] / "optim.pt", map_location=device,
                                             weights_only=True))
        start_step = int(ckpts[-1].name.split("-")[1])
        for _ in range(start_step):
            scheduler.step()
        print(f"[resume] from step-{start_step}", flush=True)

    step = start_step
    while step < total_steps:
        for ids, am, labels in loader:
            if step >= total_steps:
                break
            ids, am, labels = ids.to(device), am.to(device), labels.to(device)
            loss = model(input_ids=ids, attention_mask=am, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"[step {step}/{total_steps}] loss={float(loss):.4f}", flush=True)
            if step % 1000 == 0:
                ck = out_root / f"step-{step}"
                ck.mkdir(exist_ok=True)
                torch.save(model.state_dict(), ck / "model.pt")
                torch.save(optimizer.state_dict(), ck / "optim.pt")
                CKPTS.commit()

    best = out_root / "best"
    best.mkdir(exist_ok=True)
    model.save_pretrained(str(best))
    tok.save_pretrained(str(best))
    CKPTS.commit()
    return {"out": f"{OUT}/stage{stage}{tag}", "steps": step}


@app.function(
    gpu="A10G",
    cpu=4,
    memory=16 * 1024,
    timeout=2 * 3600,
    volumes={"/ckpts": CKPTS, "/datasets": DATA},
)
def evaluate(limit: int = 0, stage: int = 1, tag: str = "") -> dict:
    import json

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ckpt = Path("/ckpts") / OUT / f"stage{stage}{tag}" / "best"
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to("cuda").eval()

    pairs = []
    for line in (Path("/datasets/thai-ipa/test.jsonl")).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pairs.append((row["src"], row["tgt"]))
    if limit:
        pairs = pairs[:limit]

    def ed(a, b):
        prev = list(range(len(b) + 1))
        for i, ai in enumerate(a, 1):
            curr = [i]
            for j, bj in enumerate(b, 1):
                curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ai != bj)))
            prev = curr
        return prev[-1]

    total_ed = total_gold = exact = 0
    with torch.no_grad():
        for start in range(0, len(pairs), 32):
            batch = pairs[start : start + 32]
            enc = tok([s for s, _ in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            out = model.generate(**enc, max_new_tokens=256, num_beams=4)
            preds = [_decode_joined(tok, o) for o in out]
            for (_, gold), pred in zip(batch, preds, strict=True):
                e = ed(list(pred.strip()), list(gold.strip()))
                total_ed += e
                total_gold += max(1, len(gold))
                exact += e == 0
    per = round(100 * total_ed / max(1, total_gold), 2)
    exact_pct = round(100 * exact / max(1, len(pairs)), 2)
    print(f"teacher PER={per} exact={exact_pct} n={len(pairs)}", flush=True)
    return {"per": per, "n": len(pairs)}


@app.local_entrypoint()
def main() -> None:
    print(train.remote(stage=1))
    print(train.remote(stage=2))
    print(evaluate.remote())
