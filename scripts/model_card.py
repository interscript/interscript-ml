"""HF model-card generator: a models.yaml entry becomes the README.md
model card for riboseinc/<id> (the org mirror of the index id).

Curated benchmark/prose blocks live in cards/<id>.md and are wrapped
by render_card; entries without a curated block get the metrics table
alone. The index remains the single source of truth for numbers.
"""

from __future__ import annotations

HF_ORG = "interscript"

_LANG_BY_PREFIX = {
    "ara": "ar", "heb": "he", "fas": "fa", "tha": "th", "urd": "ur",
    "khm": "km",
}

_GIB = 1024 ** 3


def hf_repo_for(model_id: str) -> str:
    return f"{HF_ORG}/{model_id}"


def _fmt_size(size: int | None) -> str:
    if not size:
        return ""
    return f"{size / _GIB:.2f} GiB"


def render_card(model_id: str, entry: dict, variants: list[str],
                body: str = "", org: str = HF_ORG) -> str:
    lang = _LANG_BY_PREFIX.get(model_id.split("-")[0], "multilingual")
    lic = str(entry.get("license", "BSD-3-Clause")).lower()
    size = _fmt_size(entry.get("size"))
    lines = [
        "---",
        "license: " + lic,
        "language: " + lang,
        "inference: false",
        "tags:",
        "  - secryst",
        "  - transliteration",
    ]
    task = entry.get("task", "")
    if task == "diacritization":
        lines.append("  - diacritization")
    elif task == "g2p":
        lines.append("  - grapheme-to-phoneme")
    lines += [
        "  - onnx",
        "  - byt5",
        "---",
        "",
        f"# {model_id}",
        "",
        f"Part of the [Secryst](https://secryst.github.io) phonological-layer "
        f"family — a dedicated model for **{task}** ({lang.upper()}), distributed "
        f"as a Secryst IMF artifact and loadable through any Secryst crystal:",
        "",
        "```python",
        "from secryst import Model",
        f"Model.load(\"{model_id}\")",
        "```",
        "",
    ]
    if size:
        lines += [f"- Artifact: `{entry.get('filename', '')}` ({size})"]
    if variants:
        lines += ["- Precision variants: " + ", ".join(f"`{v}`" for v in variants)]
    metrics = entry.get("metrics") or []
    if metrics:
        lines += [
            "",
            "## Measured quality",
            "",
            "Full evaluation sets, zero skipped examples, the benchmark's own evaluator. "
            "Provenance for every number is the results record of the training project.",
            "",
            "| metric | value | provenance |",
            "|---|---|---|",
        ]
        for m in metrics:
            prov = m.get("source", "")
            lines.append(f"| {m['name']} | {m['value']} | {prov} |")
    if body:
        lines += ["", body]
    lines += [
        "",
        "## License",
        "",
        lic.upper(),
        "",
        "## Provenance",
        "",
        "Trained in the [Secryst training monorepo](https://github.com/secryst/secryst-train) "
        "and published through the [interscript-ml](https://github.com/interscript/interscript-ml) "
        "distribution contract (models.yaml index, sha256-verified release assets, "
        "parity-gated artifacts). The canonical download channel is GitHub Releases; "
        "this repository is the Hugging Face mirror of the same bytes, pinned to the "
        "index revision.",
    ]
    return "\n".join(lines) + "\n"


def ensure_repo(api, repo: str) -> None:
    """create_repo, tolerating an existing repo (org create-rights may be
    restricted while contents-write on the existing repo still works)."""
    from huggingface_hub.utils import HfHubHTTPError

    try:
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    except HfHubHTTPError as e:
        try:
            api.repo_info(repo, repo_type="model")
        except Exception:
            raise e from None
