#!/usr/bin/env bash
# DEPRECATED 2026-08-22: the npm manifest package (npm/models, formerly
# @interscript/models) is superseded by the models.yaml index consumed
# by the three secryst crystals. Kept for history; do not run.
#
# Update the npm models manifest after a release.
#
# Usage:
#   ./scripts/update_npm_manifest.sh <task> <version>
#
# Reads the asset checksums from the GH Release API and writes a new
# ``manifest.json``. Commits + publishes via the release workflow.
set -euo pipefail

TASK="${1:?task name required, e.g. rababa_arabic}"
VERSION="${2:?version required, e.g. 1.0.0}"
TAG="${TASK}-v${VERSION}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MANIFEST="npm/models/manifest.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi

python3 - "$TASK" "$VERSION" "$TAG" <<'PY'
import json, subprocess, sys
from pathlib import Path

task, version, tag = sys.argv[1:4]
manifest_path = Path("npm/models/manifest.json")
data = json.loads(manifest_path.read_text(encoding="utf-8"))

# Pull checksums from the GH Release API
api = f"repos/interscript/interscript-ml/releases/tags/{tag}"
release = json.loads(subprocess.check_output(["gh", "api", api], text=True))

assets = {}
for asset in release.get("assets", []):
    name = asset["name"]
    if name.endswith(".sha256"):
        # Download the sidecar to read the digest
        digest = subprocess.check_output(
            ["gh", "release", "download", tag, "--repo", "interscript/interscript-ml",
             "--pattern", name, "--output", "-"], text=True
        ).strip().split()[0]
        base = name[:-len(".sha256")]
        assets.setdefault(base, {})["sha256"] = digest
        assets[base]["size_bytes"] = asset["size"]
        if "-q8" in base:
            variant = "q8"
        elif "-q4" in base:
            variant = "q4"
        elif "-fp16" in base:
            variant = "fp16"
        else:
            variant = "fp32"
        assets[base]["variant"] = variant

entry = {
    "status": "active",
    "version": version,
    "released_at": release.get("created_at"),
    "assets": {k: v for k, v in assets.items()},
}
data.setdefault("models", {})[task] = {**data["models"].get(task, {}), **entry}
manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Updated {manifest_path} for {task} {version}")
PY

# Bump the npm version
(
  cd npm/models
  npm version "$VERSION" --no-git-tag-version --allow-same-version
)

echo ""
echo "Next: commit npm/models/ and publish:"
echo "  cd npm/models && npm publish"
