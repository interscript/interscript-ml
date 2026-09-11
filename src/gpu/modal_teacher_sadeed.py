"""Modal probe: teacher-only SadeedDiac-25 predictions (TODO.qwen-next/10 §2).

Runs the canonical teacher (r6 = run-006-morph) over the benchmark
under the published windowed protocol and writes per-paragraph
predictions to the checkpoints volume, so per-domain DER slicing can
be done offline (r7's slice comes from run-007's teacher column).

    modal run --detach src/gpu/modal_teacher_sadeed.py
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4,<3",
        "transformers==5.14.1",
        "numpy>=1.26",
        "pyarrow",
    )
    .add_local_dir(str(REPO_ROOT), "/root/interscript-ml", copy=True)
    .add_local_dir(
        "/Users/mulgogi/src/interscript/rababa/data/sadeed-diac-25",
        "/opt/rababa/data/sadeed-diac-25",
        copy=True,
    )
    .workdir("/root/interscript-ml")
)

CHECKPOINTS = modal.Volume.from_name("rababa-checkpoints")

TEACHER = "/checkpoints/rababa_arabic_byt5/run-006-morph/best"
OUT = "/checkpoints/probes/r6_sadeed_preds.jsonl"

app = modal.App("interscript-ml-teacher-sadeed", image=IMAGE)


@app.function(
    gpu="A10G",
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 3600,
    volumes={"/checkpoints": CHECKPOINTS},
)
def teacher_preds() -> dict:
    import json
    import sys

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Modal copies the entry file to /root/<name>.py while the repo
    # image sits at /root/interscript-ml — cover both layouts.
    for cand in (Path.cwd() / "src", Path("/root/interscript-ml/src")):
        if (cand / "harness" / "sadeed.py").exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    from harness.sadeed import strip_diacritics, windowed_paragraphs

    tok = AutoTokenizer.from_pretrained("google/byt5-small")
    teacher = AutoModelForSeq2SeqLM.from_pretrained(TEACHER).to("cuda").eval()
    teacher.generation_config.max_length = 100_000

    table = pq.read_table("/opt/rababa/data/sadeed-diac-25/train.parquet")
    inputs = [strip_diacritics(t) for t in table.column("input").to_pylist()]

    preds = windowed_paragraphs(teacher, tok, inputs, window=1400)

    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, (src, pred) in enumerate(zip(inputs, preds, strict=True)):
            f.write(
                json.dumps({"idx": i, "src": src, "teacher": pred}, ensure_ascii=False)
                + "\n"
            )
    CHECKPOINTS.commit()
    return {"rows": len(preds), "out": OUT}


@app.local_entrypoint()
def main() -> None:
    print(teacher_preds.remote())
