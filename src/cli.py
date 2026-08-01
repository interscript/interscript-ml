"""``python -m src.cli`` / ``interscript-ml`` entry point.

Subcommands:
- ``train``    fine-tune a teacher or distill a student
- ``evaluate`` run an evaluator over a saved model
- ``export``   PyTorch → ONNX
- ``publish``  upload to HuggingFace Hub
- ``list``     show registered tasks

The CLI is a thin wrapper around ``TrainingPipeline``. No business
logic here — argparse + delegation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from framework.config import load_task_config


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True, help="Task name (e.g. rababa_arabic)")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out-root", type=Path, default=Path("models"))
    p.add_argument(
        "--tasks-root",
        type=Path,
        default=None,
        help="Override tasks/ directory (default: src/tasks)",
    )


def cmd_list(args: argparse.Namespace) -> int:
    from framework.pipeline import _ensure_task_imported

    tasks_root: Path = args.tasks_root or (Path(__file__).parent / "tasks")
    if not tasks_root.is_dir():
        print(f"tasks directory not found: {tasks_root}", file=sys.stderr)
        return 1
    rows = []
    for child in sorted(tasks_root.iterdir()):
        if not (child / "config.yaml").is_file():
            continue
        try:
            cfg = load_task_config(child.name, tasks_root=tasks_root)
            _ensure_task_imported(child.name)
            rows.append(
                {
                    "task": cfg.name,
                    "kind": cfg.kind,
                    "description": cfg.description,
                    "data_module": cfg.data.module,
                    "model_module": cfg.model.module,
                    "evaluator": cfg.eval.module,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"task": child.name, "error": str(exc)})
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from framework.pipeline import TrainingPipeline

    pipeline = TrainingPipeline.from_config(
        task_name=args.task,
        data_root=args.data_root,
        out_root=args.out_root,
        tasks_root=args.tasks_root,
    )
    result = pipeline.run(max_steps=args.max_steps, skip_export=args.skip_export)
    print(
        json.dumps(
            {
                "task": result.task,
                "train_steps": result.train_steps,
                "best_loss": result.best_loss,
                "eval": result.eval.__dict__ if result.eval else None,
            },
            indent=2,
        )
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from framework.pipeline import TrainingPipeline

    pipeline = TrainingPipeline.from_config(
        task_name=args.task,
        data_root=args.data_root,
        out_root=args.out_root,
        tasks_root=args.tasks_root,
    )
    data = pipeline.build_data()
    data.prepare_data()
    evaluator = pipeline.build_evaluator()
    model = pipeline.build_model()
    predictions = []
    gold = []
    for example in data.prepared.val:
        out = model.generate([list(example.input_ids)])
        predictions.append(out.texts[0] if out.texts else "")
        gold.append(example.target)
    metric = evaluator.evaluate(predictions, gold)
    print(json.dumps({"metric": metric.__dict__}, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from framework.pipeline import TrainingPipeline

    pipeline = TrainingPipeline.from_config(
        task_name=args.task,
        data_root=args.data_root,
        out_root=args.out_root,
        tasks_root=args.tasks_root,
    )
    exporter = pipeline.build_exporter()
    if exporter is None:
        print("Task has no exporter configured", file=sys.stderr)
        return 2
    model = pipeline.build_model()
    suffix = "" if args.variant == "fp32" else f"-{args.variant}"
    out_path = args.out_root / f"{args.task}{suffix}.onnx"
    result = exporter.export(model, out_path, verify_with=model)
    print(result.format_summary())
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    repo_id = args.repo or f"interscript/{args.task}"
    print(f"Would upload {args.out_root} to huggingface.co/{repo_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interscript-ml",
        description="Unified ML training for Interscript (rababa + secryst)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available tasks")
    p_list.add_argument("--tasks-root", type=Path, default=None)
    p_list.set_defaults(func=cmd_list)

    p_train = sub.add_parser("train", help="Train a model")
    _add_common_args(p_train)
    p_train.add_argument("--max-steps", type=int, default=None)
    p_train.add_argument("--skip-export", action="store_true", default=True)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a trained model")
    _add_common_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_export = sub.add_parser("export", help="Export a trained model to ONNX")
    _add_common_args(p_export)
    p_export.add_argument(
        "--variant",
        choices=["fp32", "q8", "q4", "fp16"],
        default="fp32",
        help="Precision variant (q8/q4 produced via onnxruntime quantization)",
    )
    p_export.set_defaults(func=cmd_export)

    p_pub = sub.add_parser("publish", help="Upload to HuggingFace Hub")
    _add_common_args(p_pub)
    p_pub.add_argument("--repo", default=None)
    p_pub.set_defaults(func=cmd_publish)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
