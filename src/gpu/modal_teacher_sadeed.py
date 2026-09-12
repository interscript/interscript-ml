"""Modal probe: teacher-only SadeedDiac-25 predictions (TODO.impl/06,
the seed-soup freebie; originally TODO.qwen-next/10 §2).

Runs a teacher over the benchmark under the published windowed
protocol and writes per-paragraph predictions to the checkpoints
volume. With --soup-both, two checkpoints are weight-averaged 50/50
first (same-basin model soup; garbage output means different basins
and the probe closes negative on its own).

    modal run --detach src/gpu/modal_teacher_sadeed.py
    modal run --detach src/gpu/modal_teacher_sadeed.py --soup-both
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
SOUP_PARTNER = "/checkpoints/rababa_arabic_byt5/run-007-news/best"
OUT = "/checkpoints/probes/r6_sadeed_preds.jsonl"
OUT_SOUP = "/checkpoints/probes/r67_soup_sadeed_preds.jsonl"

app = modal.App("interscript-ml-teacher-sadeed", image=IMAGE)


@app.function(
    gpu="A10G",
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 3600,
    volumes={"/checkpoints": CHECKPOINTS},
)
def teacher_preds(soup_both: bool = False) -> dict:
    import json
    import sys

    import pyarrow.parquet as pq
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Modal copies the entry file to /root/<name>.py while the repo
    # image sits at /root/interscript-ml — cover both layouts.
    for cand in (Path.cwd() / "src", Path("/root/interscript-ml/src")):
        if (cand / "harness" / "sadeed.py").exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    from harness.sadeed import strip_diacritics, windowed_paragraphs

    tok = AutoTokenizer.from_pretrained("google/byt5-small")
    teacher_path = SOUP_PARTNER if soup_both else TEACHER
    out_path = OUT_SOUP if soup_both else OUT
    teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher_path)
    if soup_both:
        partner = AutoModelForSeq2SeqLM.from_pretrained(TEACHER)
        soup = {
            name: (teacher.state_dict()[name] + partner.state_dict()[name]) / 2
            for name in teacher.state_dict()
        }
        del partner
        teacher.load_state_dict(soup)
        del soup
    teacher = teacher.to("cuda").eval()
    teacher.generation_config.max_length = 100_000

    table = pq.read_table("/opt/rababa/data/sadeed-diac-25/train.parquet")
    inputs = [strip_diacritics(t) for t in table.column("input").to_pylist()]

    preds = windowed_paragraphs(teacher, tok, inputs, window=1400)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, (src, pred) in enumerate(zip(inputs, preds, strict=True)):
            f.write(
                json.dumps({"idx": i, "src": src, "teacher": pred}, ensure_ascii=False)
                + "\n"
            )
    CHECKPOINTS.commit()
    return {"rows": len(preds), "out": str(out_path), "souped": soup_both}


@app.local_entrypoint()
def main(soup_both: bool = False) -> None:
    print(teacher_preds.remote(soup_both=soup_both))
