# interscript-ml contract — Specification (v1)

Normative. Key words MUST / MUST NOT / SHALL / SHOULD / MAY are to be
interpreted as described in RFC 2119. This file is the canonical text;
the [crystals' documentation site](https://www.secryst.org/spec.html)
renders the same content for users.

## Conformance

An implementation conforms to interscript-ml v1 when it:

- **C1** — MUST resolve model ids against a `models.yaml` index of
  `version: 1`, honoring `SECRYST_INDEX`.
- **C2** — MUST verify the whole-artifact sha256 before installation
  and re-verify every cache hit; a mismatch MUST fail loudly.
- **C3** — MUST verify every `.onnx` member against the manifest
  sha256 map on load; members not covered by the manifest MUST NOT
  load.
- **C4** — MUST implement the byte tokenizer exactly (§4); id
  sequences MUST NOT be treated as raw bytes.
- **C5** — MUST produce byte-identical outputs to the reference
  crystal on the shared golden sets.
- **C6** — SHOULD install artifacts atomically (temp file + rename).

## §1 The models.yaml index

    version: 1
    models:
      <id>:
        filename: string          # artifact file name
        url: string               # single-file channel (http(s):// or file://)
        sha256: string            # whole-artifact digest
        size: int
        precision: fp32 | fp16    # default fp32
        task: string
        parts:                    # OPTIONAL: split artifacts
          - url: string
            sha256: string        # per-part digest, verified as it lands
            size: int

The `parts` mechanism exists for artifacts exceeding GitHub's 2 GiB
per-asset cap: parts stream into one file in index order, each verified
on arrival; the assembled file is then checked against the entry-level
`sha256` exactly as a single-file model — the cache contract is
identical.

## §2 Resolution algorithm

1. Resolve `<id>` in `models.models`. Unknown ids MUST raise an error
   enumerating known ids.
2. If a cached copy exists at `<cache>/models/<id>/<filename>` whose
   whole-file sha256 matches, use it — cache hits are re-verified,
   never trusted blindly.
3. Otherwise download (single URL, or parts in order) to a temporary
   file in the target directory, verifying digests as data lands.
4. Verify the assembled artifact against the index sha256.
5. Atomically rename into place, then load.

## §3 IMF v1 model zips

A zip containing at minimum `metadata.yaml`, `encoder.onnx`, and
`decoder.onnx`. Zips MUST NOT rely on zip-level integrity; integrity
is the manifest's job.

    format: imf-v1
    tokenizer: bytes
    id: <model id>
    task: string
    decoder: plain | kv        # kv iff decoder-kv.onnx is present
    precision: fp32
    opset: 14
    sha256:                     # every .onnx member MUST be covered
      encoder.onnx: <hex>
      decoder.onnx: <hex>

A conforming loader MUST reject: any other `format`, any `tokenizer`
other than `bytes`, missing required members, and uncovered or
mismatched member digests.

## §4 Byte tokenizer

| concept | rule |
|---|---|
| encoding a string | UTF-8 bytes `b` → ids `b + 3`, then one trailing `eos` |
| decoding ids | stop at `eos`; skip `pad`/`unk`; `(id − 3) mod 256` per byte; reassemble as UTF-8 |
| special ids | `pad = 0`, `eos = 1`, `unk = 2` |

Warning — the classic silent-garbage bug: ids are offset by 3 and carry
a trailing EOS. Feeding `text.bytes` directly, or forgetting the EOS,
produces plausible-but-wrong outputs that pass shape checks. Interop
tests MUST cover both encode and decode round-trips, including
multi-byte scripts.

## §5 Environment

| variable | meaning | default |
|---|---|---|
| `SECRYST_INDEX` | index URL or local path | `models.yaml` on this repo's main |
| `SECRYST_CACHE` | artifact cache directory | `~/.cache/secryst` |

## Conformance kits

- Golden parity kit (deterministic decode-loop fixture + reference
  goldens): [secryst-py/parity](https://github.com/secryst/secryst-py/tree/main/parity).
- Export gate (parity written into released artifacts): this repo's
  release pipeline (`src/imf/`, WO03).
