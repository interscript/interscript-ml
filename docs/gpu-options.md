# GPU options — where to train

We need a GPU for the heavy lifting (teacher fine-tune + distillation).
The framework's CPU path works for dev / CI / mobile variants, but
production training needs CUDA. Here are the realistic options.

## Cost vs need

| Task | Wall time on 1× A100 80GB | Est. cost (Modal A100) |
|---|---|---|
| rababa_arabic teacher fine-tune (Qwen3.5-4B LoRA) | 12-18h | ~$15-25 |
| rababa_arabic student distillation | 3-4h | ~$5-7 |
| rababa_arabic ONNX export + benchmarks (CPU) | 5min | $0 |
| **rababa_arabic total** | **~16-22h** | **~$20-30** |
| rababa_hebrew (same shape) | ~16-22h | ~$20-30 |
| secryst_thai_ipa (same shape) | ~16-22h | ~$20-30 |
| **All three tasks** | **~50-66h** | **~$60-90** |

For comparison: $100 of Modal credits is enough to retrain every task
3-4 times. **Cost is not the blocker.**

## Recommendation

### Tier 1: Modal (default for dev) — serverless A10G/A100

```bash
pip install modal
modal token new
modal run src/gpu/modal_train.py --task rababa_arabic
```

**Why:**
- Per-second billing (no minimum)
- No queue, no commitment
- Image is baked; cold start ~30s
- Mounts local code → fast iteration
- A10G ($1.09/hr) is enough for our sizes
- A100 ($3.40/hr) only needed for the 4B teacher

**Code:** `src/gpu/modal_train.py` — already written.

### Tier 2: Lambda Labs (sustained) — A100 80GB at $1.10/hr

Use when retraining all three tasks back-to-back. 50h × $1.10 = $55.
Same hardware as AWS/GCP, half the price, simpler billing. Trade-off:
queues can be hours; reserved instances take a week to provision.

### Tier 3: Colab Free (zero budget) — T4 16GB

T4 fits the student distillation (6M params). Teacher (4B + LoRA) is
tight — needs gradient checkpointing + 4-bit base model loading.

**Workflow:**
1. Open `notebooks/colab_train.ipynb` in Colab Free
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Model auto-uploads to HF Hub at end

Limitations:
- 12h session limit (rababa_arabic teacher fits in 12h)
- T4 is older architecture — 2x slower than A10G
- Storage is ephemeral — must checkpoint to Drive or HF

### Tier 4: HuggingFace Spaces A10G Small (free)

For inference + small jobs only. Not suitable for training (1 GPU,
shared, preemptible). Use for the demo deployment instead.

### Tier 5: AWS / GCP / Azure (sponsored)

Apply for OSS credits. Interscript qualifies for:
- **AWS Open Source Software Sponsorship** ($1-5k credits)
- **GCP for Open Source** ($5-25k credits via Google for Startups OSS)
- **HuggingFace Community Grants** (free compute for OSS ML)
- **NumFOCUS Small Grants** ($3-5k for fiscal-sponsored projects)
- **MLCommons Research Credits** (academic partnerships)

A single AWS sponsorship would cover all training for 12+ months at
our scale. Apply early — these take weeks.

## Decision matrix

| Use case | Recommended |
|---|---|
| First production run | Modal A100 ($20-30/task) |
| Dev iteration | Local CPU (this repo) or Colab T4 |
| Bulk retrain all tasks | Lambda Labs A100 |
| Cheap / free demo | Colab Free T4 |
| Long-term | AWS/GCP OSS credits |
| Mobile/edge variants | Local CPU (no GPU needed) |

## Anti-choices

- **Kaggle Kernels**: 30h/week, but kernel restarts lose progress;
  better for one-off demos than sustained work.
- **Vast.ai**: cheapest but reliability issues; intermittent driver
  crashes on consumer cards.
- **RunPod serverless**: similar to Modal but smaller community; only
  use if Modal raises prices.
- **Local GPU**: only if you already have an A6000 or better. RTX 3090
  is borderline — fits student but not teacher.

## Monitoring cost

Modal dashboard shows live cost per run. Set a hard ceiling via:

```python
@stub.function(gpu="A100", timeout=6 * 3600, cpu=8, memory=32 * 1024)
def train_task(task: str):
    # Modal aborts if wall time exceeds timeout — no runaway bill.
    ...
```

Plus: set `INTERSCRIPT_MAX_USD_PER_RUN` env var (TODO in the framework)
to abort if cumulative cost crosses a threshold.
