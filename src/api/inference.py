"""IMF v1 inference endpoint for api.interscript.org (Modal, CPU).

Serves the shipped models from the secryst-models volume using the
exact ONNX kv decode the WO03 parity gate verified. Cold start loads
the fp32 zip (~30-60s); sessions are cached per container.

    modal deploy src/api/inference.py

Auth: X-API-Key header must match the `api-inference-key` secret.
"""

from pathlib import Path

import modal

models_volume = modal.Volume.from_name("secryst-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("onnxruntime==1.23.2", "pyyaml>=6.0", "fastapi>=0.115")
    .add_local_dir(str(Path(__file__).resolve().parent.parent), "/root/interscript-ml", copy=True)
    .workdir("/root/interscript-ml")
    .env({"IMAGE_REV": "5"})
)

app = modal.App("interscript-inference", image=image)

MAX_INPUT_BYTES = 4000
MAX_OUTPUT_TOKENS = 8192
ALLOWED_TASKS = ("diacritization", "g2p")

_sessions: dict[str, tuple] = {}


def _zip_path(model_id: str) -> Path:
    family = model_id.rsplit("-", 1)[0]
    p = Path("/v/imf") / family / f"{model_id}-fp32.zip"
    if not p.exists():
        raise KeyError(model_id)
    return p


def _get_sessions(model_id: str) -> tuple:
    import sys

    sys.path.insert(0, "/root/interscript-ml/src")
    from imf.parity import _sessions_from_zip  # the parity-verified loader

    if model_id not in _sessions:
        _sessions[model_id] = _sessions_from_zip(_zip_path(model_id))
    return _sessions[model_id]


def _metadata(model_id: str) -> dict:
    import zipfile

    import yaml

    with zipfile.ZipFile(_zip_path(model_id)) as zf:
        return yaml.safe_load(zf.read("metadata.yaml"))


def _decode(tokens: list) -> str:
    # ByT5 token ids are byte+3, with trailing EOS (id 1)
    return bytes(t - 3 for t in tokens if t >= 3).decode("utf-8", "replace")


def make_api():
    import os
    from fastapi import FastAPI, HTTPException, Request

    from pydantic import BaseModel

    api = FastAPI(title="Interscript inference", version="1.0.0")

    @api.post("/infer")
    async def infer(request: Request) -> dict:
        import sys

        key = request.headers.get("x-api-key", "")
        if not key or key != os.environ.get("API_INFERENCE_KEY"):
            raise HTTPException(401, "invalid or missing X-API-Key")
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "body must be JSON {model, input}") from None
        if not isinstance(body, dict) or not body.get("model") or not body.get("input"):
            raise HTTPException(400, "body must be {model, input}")

        return await _run_infer(body)

    async def _run_infer(body):
        import sys

        models_volume.reload()
        try:
            meta = _metadata(body["model"])
        except KeyError:
            raise HTTPException(404, f"unknown model {body['model']}") from None
        if meta.get("task") not in ALLOWED_TASKS:
            raise HTTPException(400, f"model {body['model']} task {meta.get('task')} is not served")

        if len(body["input"].encode("utf-8")) > MAX_INPUT_BYTES:
            raise HTTPException(413, f"input exceeds {MAX_INPUT_BYTES} bytes")

        sys.path.insert(0, "/root/interscript-ml/src")
        from imf.export import onnx_greedy_kv

        enc, kv = _get_sessions(body["model"])
        max_len = min(MAX_OUTPUT_TOKENS, 3 * len(body["input"].encode("utf-8")) + 256)
        output = _decode(onnx_greedy_kv(enc, kv, body["input"], max_len))
        return {
            "model": body["model"],
            "task": meta["task"],
            "source_script": meta.get("source_script"),
            "input": body["input"],
            "output": output,
        }

    @api.get("/health")
    def health() -> dict:
        import glob

        models_volume.reload()
        return {"ok": True, "models": len(glob.glob("/v/imf/*/*-fp32.zip"))}

    return api


@app.function(
    cpu=4,
    memory=8 * 1024,
    timeout=10 * 60,
    volumes={"/v": models_volume},
    secrets=[modal.Secret.from_name("api-inference-key")],
    # cold starts (~30-60s model load) instead of a 24/7 warm container
    # — money discipline; bump once traffic justifies it
)
@modal.asgi_app()
def web():
    return make_api()
