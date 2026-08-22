#!/usr/bin/env python3
"""Publish an IMF zip: validate -> (split if >2GiB) -> GH Release ->
models.yaml entry -> PR. One command, the whole tail of the pipeline.

    python scripts/publish_model.py heb-diac-1.0 --zip /tmp/heb-diac-1.0-fp32.zip

Idempotent: an existing release gets assets re-uploaded (clobber), an
existing models.yaml entry block is replaced in place, an existing
release branch/PR is reused. Never touches main directly.

The metadata inside the zip is the source of truth for metrics/parity;
the strict validator gate must pass before anything is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from imf.validator import validate_zip  # noqa: E402
from split_release import split as split_zip  # noqa: E402

# GitHub hard-caps release assets at 2,147,483,648 bytes; split well below.
SPLIT_THRESHOLD = 2_000_000_000
DEFAULT_REPO = "interscript/interscript-ml"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return yaml.safe_load(zf.read("metadata.yaml"))


def entry_block(model_id: str, meta: dict, filename: str, sha256: str, size: int,
                assets: list[Path], repo: str, tag: str) -> str:
    base = f"https://github.com/{repo}/releases/download/{tag}"
    lines = [
        f"  {model_id}:",
        f"    task: {meta['task']}",
        f"    scripts: [{meta['source_script']}, {meta['target']}]",
        f"    precision: {meta['precision']}",
        f"    filename: {filename}",
    ]
    if len(assets) == 1:
        lines.append(f"    url: {base}/{assets[0].name}")
    else:
        lines.append("    parts:")
        for asset in assets:
            lines += [
                f"      - url: {base}/{asset.name}",
                f"        sha256: {sha256_file(asset)}",
                f"        size: {asset.stat().st_size}",
            ]
    lines += [
        f"    sha256: {sha256}",
        f"    size: {size}",
        "    metrics:",
    ]
    for metric in meta["metrics"]:
        lines.append(
            f"      - {{name: {metric['name']}, value: {metric['value']}, "
            f"source: {metric['source']}}}"
        )
    parity = meta["parity"]
    lines += [
        f"    parity: {{samples: {parity['samples']}, cer_delta: {parity['cer_delta']}}}",
        f"    license: {meta['license']}",
    ]
    return "\n".join(lines) + "\n"


def upsert_models_yaml(models_yaml: Path, model_id: str, block: str) -> None:
    text = models_yaml.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^  {re.escape(model_id)}:\n(?:(?!^  \S).*\n)*", re.MULTILINE
    )
    if pattern.search(text):
        models_yaml.write_text(pattern.sub(block, text), encoding="utf-8")
    else:
        with models_yaml.open("a", encoding="utf-8") as fh:
            fh.write(block)


def release_notes(model_id: str, meta: dict, filename: str, size: int,
                  sha256: str, assets: list[Path]) -> str:
    lines = [
        f"# {model_id}",
        "",
        f"IMF v1 (`{meta['precision']}`, decoder `{meta['decoder']}`, opset "
        f"{meta['opset']}). Trained from {meta['trained_from']}.",
        "",
        "| field | value |",
        "|---|---|",
        f"| task | {meta['task']} ({meta['source_script']} → {meta['target']}) |",
        f"| artifact | {filename} ({size / 1024**3:.2f} GiB)",
    ]
    if len(assets) > 1:
        lines.append(f"| parts | {' + '.join(a.name for a in assets)} (GitHub 2GiB cap) |")
    for metric in meta["metrics"]:
        lines.append(f"| {metric['name']} | {metric['value']} — {metric['protocol']} |")
    parity = meta["parity"]
    lines += [
        f"| parity | cer_delta {parity['cer_delta']}pp on {parity['samples']} samples |",
        f"| sha256 | `{sha256}` |",
        f"| license | {meta['license']} |",
        "",
        "Runtimes reassemble split parts transparently and verify every sha256:",
        "",
        "```python",
        "from secryst import Model",
        f'model = Model.load("{model_id}")',
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()

    # git and gh must operate on this repo regardless of the caller's cwd
    os.chdir(REPO_ROOT)

    if args.zip.resolve().parent == REPO_ROOT:
        raise SystemExit("refusing to publish from the repo root; keep zips outside the tree")

    result = validate_zip(args.zip, strict=True)
    if not result.ok:
        result.errors and print("\n".join(result.errors), file=sys.stderr)
        raise SystemExit("strict validation failed; nothing published")

    meta = load_metadata(args.zip)
    if meta["id"] != args.model_id:
        raise SystemExit(f"metadata id {meta['id']!r} != requested {args.model_id!r}")

    whole_sha = sha256_file(args.zip)
    size = args.zip.stat().st_size

    assets = [args.zip]
    if size > SPLIT_THRESHOLD:
        print(f"{size:,} bytes > {SPLIT_THRESHOLD:,}; splitting for the GitHub asset cap")
        parts = split_zip(args.zip, 1_500_000_000)
        assets = [part for part, _, _ in parts]

    tag = args.model_id
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(release_notes(args.model_id, meta, args.zip.name, size, whole_sha, assets))
        notes_path = fh.name

    existing = subprocess.run(
        ["gh", "release", "view", tag, "--json", "name"],
        capture_output=True, text=True,
    )
    if existing.returncode == 0:
        listed = run(["gh", "release", "view", tag, "--json", "assets",
                      "--jq", "[.assets[] | {name, size}]"],
                     capture_output=True, text=True).stdout
        have = {
            asset["name"]: int(asset["size"])
            for asset in yaml.safe_load(listed)
        }
        want = {a.name: a.stat().st_size for a in assets}
        if have == want:
            print(f"release {tag} already carries all assets; skipping upload")
        else:
            print(f"release {tag} exists; re-uploading assets (clobber)")
            run(["gh", "release", "upload", tag, *[str(a) for a in assets], "--clobber"])
        run(["gh", "release", "edit", tag, "--notes-file", notes_path])
        # a create that died mid-upload leaves the release as a draft;
        # the re-run must publish it
        is_draft = run(["gh", "release", "view", tag, "--json", "isDraft"],
                       capture_output=True, text=True).stdout
        if yaml.safe_load(is_draft):
            run(["gh", "release", "edit", tag, "--draft=false"])
    else:
        run(["gh", "release", "create", tag, *[str(a) for a in assets],
             "--title", tag, "--notes-file", notes_path])

    # Branch work happens in a dedicated worktree so the caller's tree is
    # never checked out (dirty files must not block publication, and
    # publication must not clobber in-progress edits).
    branch = f"release/{args.model_id}"
    worktree = REPO_ROOT.parent / f".wt-publish-{args.model_id}"
    branches = run(["git", "branch", "--list", branch],
                   capture_output=True, text=True).stdout
    base = branch if branches else "origin/main"
    run(["git", "worktree", "add", str(worktree), "-B", branch, base])

    try:
        wt_models = worktree / "models.yaml"
        wt_repo = str(worktree)
        # models/<family>/<id>.metadata.yaml, e.g. models/heb-diac/heb-diac-1.0...
        family = args.model_id.rsplit("-", 1)[0]
        upsert_models_yaml(wt_models, args.model_id,
                           entry_block(args.model_id, meta, args.zip.name, whole_sha,
                                       size, assets, args.repo, tag))
        model_dir = worktree / "models" / family
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"{args.model_id}.metadata.yaml").write_text(
            yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")

        def git_wt(*cmd: str):
            return run(["git", "-C", wt_repo, *cmd], capture_output=True, text=True)

        git_wt("add", "models.yaml", f"models/{family}/{args.model_id}.metadata.yaml")
        staged = git_wt("diff", "--cached", "--name-only").stdout.split()
        if staged:
            git_wt("commit", "-m", f"release: {args.model_id} "
                   f"({meta['precision']}, parity cer_delta {meta['parity']['cer_delta']}pp "
                   f"on {meta['parity']['samples']} samples)")
            git_wt("push", "-u", "origin", branch)
        else:
            print("nothing new to commit (idempotent re-run)")
        prs = run(["gh", "pr", "list", "--head", branch, "--json", "number"],
                  capture_output=True, text=True).stdout
        if not yaml.safe_load(prs):
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
                fh.write(f"## Summary\n- publish {args.model_id} ({meta['precision']}): "
                         f"GH Release `{tag}` + models.yaml entry"
                         f"{' (split parts, GitHub 2GiB cap)' if len(assets) > 1 else ''}\n"
                         f"- parity cer_delta {meta['parity']['cer_delta']}pp on "
                         f"{meta['parity']['samples']} samples; strict validator gate passed\n\n"
                         f"## Test plan\n- [ ] CI green\n- [ ] runtime fetch "
                         f"`Model.load(\"{args.model_id}\")` resolves and verifies\n")
                body_path = fh.name
            # the branch push propagates asynchronously; an immediate
            # pr create reliably fails with "head branch not found"
            import time

            for attempt in range(3):
                proc = subprocess.run(
                    ["gh", "pr", "create", "-R", args.repo,
                     "--head", branch, "--title", f"release: {args.model_id}",
                     "--body-file", body_path],
                    capture_output=True, text=True,
                )
                if proc.returncode == 0:
                    break
                print(f"pr create attempt {attempt + 1} failed: {proc.stderr.strip()}")
                time.sleep(20)
            else:
                raise SystemExit("gh pr create failed after retries")
    finally:
        run(["git", "worktree", "remove", "--force", str(worktree)])
    print(f"published {args.model_id}: release {tag}, branch {branch}")


if __name__ == "__main__":
    main()
