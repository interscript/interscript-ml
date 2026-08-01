"""GPU cloud training integration.

Wraps the framework's ``TrainingPipeline.run`` for cloud GPU runtimes.
Currently supports Modal (serverless, per-second billing). Adding
RunPod/Lambda Labs = one new file each (OCP).
"""
