# 07 — CI/CD for releases

**Status:** SPECIFICATION
**Priority:** P2

## Goal

`git tag <task>-v<x.y.z> && git push --tags` produces:
1. ONNX export (fp32, q8, q4)
2. SHA256 checksums
3. Benchmarks (DER/PER, latency)
4. GH Release with auto-generated notes
5. HF Hub upload (PyTorch + ONNX + card)
6. `npm `secryst` (manifest now the models.yaml index)` npm manifest bump
7. (Optional) `@interscript/model-<task>` npm publish

Zero manual steps after the tag.

## Workflow files

### `.github/workflows/test.yml` (on every PR)

```yaml
name: test
on: [push, pull_request]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install -e ".[dev]"
      - run: pytest --cov=src
      - run: ruff check src tests
      - run: mypy src
```

### `.github/workflows/release.yml` (on tag push)

```yaml
name: release
on:
  push:
    tags: ["*-v*.*.*"]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        variant: [fp32, q8, q4]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install -e ".[train,export]"
      - name: Parse tag
        id: tag
        run: |
          echo "task=$(echo $GITHUB_REF_NAME | cut -d- -f1-2)" >> $GITHUB_OUTPUT
          echo "version=$(echo $GITHUB_REF_NAME | cut -d- -f3)" >> $GITHUB_OUTPUT
      - name: Export ONNX
        run: |
          python -m src.cli export \
            --task ${{ steps.tag.outputs.task }} \
            --variant ${{ matrix.variant }} \
            --out-root models/${{ steps.tag.outputs.task }}
      - uses: actions/upload-artifact@v4
        with:
          name: onnx-${{ matrix.variant }}
          path: models/${{ steps.tag.outputs.task }}/*.onnx

  benchmarks:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with: { name: onnx-fp32, path: models }
      - run: pip install -e ".[export]"
      - name: Run benchmark suite
        run: |
          python -m src.cli evaluate \
            --task ${{ steps.tag.outputs.task }} \
            --data-root data \
            --out-root models \
            > benchmarks.json
      - uses: actions/upload-artifact@v4
        with:
          name: benchmarks
          path: benchmarks.json

  release:
    needs: [build, benchmarks]
    runs-on: ubuntu-latest
    permissions: { contents: write }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
      - name: Compute checksums
        run: |
          for f in models/*.onnx; do
            sha256sum "$f" > "$f.sha256"
          done
      - name: Generate release notes
        id: notes
        run: python scripts/generate_release_notes.py --task ${{ steps.tag.outputs.task }}
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          body_path: RELEASE_NOTES.md
          files: |
            models/*.onnx
            models/*.onnx.sha256
            models/*-vocab.json
            models/*-config.json
            benchmarks.json

  publish-hf:
    needs: release
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
      - run: pip install -e ".[publish]"
      - env: { HF_TOKEN: "${{ secrets.HF_TOKEN }}" }
        run: |
          python -m src.cli publish \
            --task ${{ steps.tag.outputs.task }} \
            --repo interscript/${{ steps.tag.outputs.task }}

  publish-npm:
    needs: release
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20", registry-url: "https://registry.npmjs.org" }
      - run: ./scripts/update_npm_manifest.sh ${{ steps.tag.outputs.task }} ${{ steps.tag.outputs.version }}
      - working-directory: npm/models
        run: npm publish
        env: { NODE_AUTH_TOKEN: "${{ secrets.NPM_TOKEN }}" }
```

## Idempotency

Re-running the workflow on an existing tag is a no-op:
- GH Release `softprops/action-gh-release@v2` is idempotent (creates or updates).
- HF Hub upload overwrites same-version files.
- npm publish on same version fails (intentional — patch-bump instead).

## Failure handling

If any step fails:
- GH Release is NOT created (atomic).
- HF upload is NOT triggered.
- npm publish is NOT triggered.
- Tag remains (manual investigation).
- Slack/email notification via `slack-notification` action.

## Local reproduction

`scripts/release_local.sh <task>` reproduces the CI flow on a laptop:

```bash
./scripts/release_local.sh rababa_arabic
# → exports ONNX (fp32 + q8)
# → computes SHA256
# → runs benchmark
# → prints release notes draft
# → DOES NOT publish (review first)
```

This lets you preview a release before tagging.

## Acceptance

- [ ] `.github/workflows/test.yml` runs on PRs
- [ ] `.github/workflows/release.yml` runs on tags
- [ ] Local reproduction script works
- [ ] Failed release does not leave partial state
- [ ] First end-to-end dry-run on `rababa_arabic-v0.1.0-alpha.1`
