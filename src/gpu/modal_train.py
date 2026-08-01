"""Modal.com integration — serverless GPU training.

Run a full training job on Modal with one command:

    modal run src/gpu/modal_train.py --task rababa_arabic

Cost: ~$1/hr for A10G, ~$3/hr for A100. Per-task training fits in
1-6 hours, so $1-$20 per task total.

Setup (one-time):

    pip install modal
    modal token new

The Modal stub image bakes in the framework + train extras + a pinned
torch+cu121 wheel. Code is mounted from the local checkout at run time,
so framework edits are picked up without rebuilding the image.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import modal  # type: ignore
    _MODAL_AVAILABLE = True
except ImportError:
    modal = None  # type: ignore
    _MODAL_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GPU_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.2.0",
        "transformers>=4.42",
        "datasets>=2.18",
        "accelerate>=0.27",
        "peft>=0.10",
        "sentencepiece>=0.2",
        "onnx>=1.16",
        "onnxruntime>=1.17",
        "onnxscript>=0.0.7",
        "pyyaml>=6.0",
        "pydantic>=2.5",
        "huggingface_hub>=0.23",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .copy_directory(str(REPO_ROOT), "/root/ml-models")
    .workdir("/root/ml-models")
    .run_commands("pip install -e '.[dev]'")
)

stub = modal.Stub("interscript-ml-train", image=GPU_IMAGE) if _MODAL_AVAILABLE else None


if _MODAL_AVAILABLE:

    @stub.function(gpu="A10G", timeout=6 * 3600, cpu=4, memory=16 * 1024)
    def train_task(task: str, max_steps: int | None = None) -> dict:
        """Train one task on Modal. Returns the pipeline result as a dict."""
        import sys
        sys.path.insert(0, "/root/ml-models/src")
        from framework.pipeline import TrainingPipeline

        pipeline = TrainingPipeline.from_config(
            task_name=task,
            data_root=Path("/root/ml-models/data"),
            out_root=Path("/root/ml-models/models") / task,
        )
        result = pipeline.run(max_steps=max_steps, skip_export=False)
        return {
            "task": result.task,
            "train_steps": result.train_steps,
            "best_loss": result.best_loss,
            "eval": result.eval.__dict__ if result.eval else None,
            "export_path": str(result.export.path) if result.export else None,
        }

    @stub.local_entrypoint()
    def main(task: str, max_steps: int = 0):
        """Local entrypoint invoked by `modal run`."""
        result = train_task.remote(task, max_steps or None)
        print(result)


if __name__ == "__main__" and not _MODAL_AVAILABLE:
    print(
        "modal is not installed. Install with: pip install modal\n"
        "Then: modal token new\n"
        "Then: modal run src/gpu/modal_train.py --task rababa_arabic",
        file=sys.stderr,
    )
    sys.exit(1)
