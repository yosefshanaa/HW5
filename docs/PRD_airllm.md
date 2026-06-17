# PRD — Mechanism: AirLLM Layer-by-Layer Execution

| Field | Value |
|---|---|
| **Mechanism** | AirLLM layer-streaming inference (the core of the assignment) |
| **Version** | 1.00 · **Updated** 2026-06-17 |
| **Parent** | [PRD.md](./PRD.md) · **Design** [PLAN.md](./PLAN.md) |
| **Owning module** | `services/airllm_runner.py` + `sdk/model_loader/airllm_backend.py` |

---

## 1. What it is & why

A 70B FP16 model is **~140 GB**; this machine has **7.6 GiB RAM and no CUDA GPU**. Normally impossible. **AirLLM changes the point of view** (Part-C p.3): don't compress the whole model into memory — **change the way the model is *loaded***.

> *"More performance on existing hardware — because the bottleneck is now **time**, not **memory**."* (Part-C p.3)

**One token, layer by layer** (Part-C p.4–5):
1. Load **one** transformer layer's weights from disk → VRAM/RAM.
2. Run the forward pass for that layer; produce a small **hidden state**.
3. **Release** the layer; load the next.
4. Repeat for all N layers → next token. Peak memory = **one layer**, not the model.

The weights are stored as **per-layer `safetensors` shards** and read via **`mmap`** (Part-C p.7–8): the OS maps the file into virtual address space and a **page fault** pulls in only the pages actually touched; the **page cache** lets repeated reads reuse already-loaded pages. The cost: weights are re-read from disk **every token** → the bottleneck **shifts from VRAM to I/O & latency** (Part-C p.6, p.9).

---

## 2. Scope on *this* hardware (critical)

- **No CUDA** → run AirLLM on **CPU** (`device="cpu"`). The "Layer Load → GPU compute → release" cycle becomes "Layer Load → **CPU** compute → release." The systems story is identical; only the compute engine and speed differ.
- **`/mnt/c` is 9p (slow)** → disk reads are *worse* than native NVMe, so AirLLM's I/O penalty is **amplified**. We deliberately measure this and also test WSL2-native ext4 (`~`) for contrast (extension E1 / ADR-003).
- This is the honest worst case for AirLLM and therefore the most informative.

---

## 3. Functional requirements (this mechanism)

| ID | Requirement | Acceptance |
|---|---|---|
| **AF-1** | Load via **`AutoModel.from_pretrained(model_id)`** (Part-C p.19), **never** a model-specific class | avoids Qwen/other class-mismatch (exercise §5.3) |
| **AF-2** | Set **`layer_shards_saving_path`** to a configurable dir with free space, **not** `C:\` root | shards land there; `C:` not flooded (ADR-003) |
| **AF-3** | Cache layout per Part-C p.14 (`…/<model>/layer_NN/…`, `tokenizer/`, `config/`, `logs/`) | inspectable on disk; reused across runs |
| **AF-4** | Generation uses **low `max_new_tokens`** (16–32) and short prompts | "Start Small" (Part-C p.19); run completes |
| **AF-5** | Device detected at runtime; CPU path forced here | `torch.cuda.is_available()==False` handled |
| **AF-6** | Tokenizer pad/eos configured to avoid the common tokenizer error | no padding/eos crash (Part-C p.21) |
| **AF-7** | Per-layer **load-time and compute-time** instrumented | timeline data emitted (feeds FR-17) |
| **AF-8** | HF token (for gated models) via **gatekeeper/env only** | never hard-coded (Part-C p.20) |
| **AF-9** | Optional 4bit/8bit **compression** treated as an *optimization lever*, not a precondition | FP16 runs without quant (Part-C p.12); see [PRD_quantization](./PRD_quantization.md) |
| **AF-10** | Prefetch/overlap option explored (load layer *N+1* during compute of *N*) | extension E2; measure I/O-wait reduction (Part-C p.13) |

---

## 4. Reference integration (sketch)

```python
# sdk/model_loader/airllm_backend.py  (≤150 lines; illustrative)
from airllm import AutoModel
from transformers import AutoTokenizer

def load(cfg, gatekeeper):
    token = gatekeeper.hf_token()                       # AF-8: env only
    model = AutoModel.from_pretrained(                  # AF-1: AutoModel
        cfg.model_id,
        layer_shards_saving_path=cfg.shards_path,       # AF-2/3: off C:
        # compression=cfg.compression,                  # AF-9: optional 4bit/8bit
        hf_token=token,
    )
    tok = AutoTokenizer.from_pretrained(cfg.model_id, token=token)
    if tok.pad_token is None:                            # AF-6
        tok.pad_token = tok.eos_token
    return model, tok

def generate(model, tok, prompt, max_new_tokens=32):    # AF-4
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)
```

---

## 5. Theory linkage (for the report — FR-22)

- **Memory→Time trade (Part-C p.9):** AirLLM lowers peak memory by re-reading weights each step → **higher latency**. We *replaced a memory limit with a time limit.*
- **Decode is memory/I/O-bound (Part-A roofline / Memory Wall):** the decode phase already re-reads weights per token; AirLLM makes that read come **from disk**, so per-layer **I/O Wait** dominates the timeline (Part-C p.9, p.11).
- **`mmap` + page fault + page cache (Part-B, Part-C p.7):** explains *why* the first (cold) run is slow and repeat (warm) runs speed up — the OS page cache retains recently used shards. Quantified by extension E3.
- **Prefetching (Part-C p.13):** overlapping next-layer load with current-layer compute hides part of the I/O wait — "don't wait for compute to finish before starting the next read."

---

## 6. Where AirLLM fits (and doesn't)

**Fits (Part-C p.10/16):** research, POC, quality evaluation, accessing a giant model when **VRAM access matters more than speed**, on limited hardware.
**Doesn't (Part-C p.11):** production high-throughput, real-time chat, many concurrent users, low-latency — there, prefer a small/quantized model on Ollama, full-VRAM GPU, or a dedicated server. This bounds NG-1 in the parent PRD.

**Decision map (Part-C p.16):** Enough VRAM? → Ollama/standard. Else need a giant model? → **AirLLM layer-loading**. Else → smaller quantized model.

---

## 7. Risks (mechanism-specific)

| Risk | Mitigation |
|---|---|
| 70B CPU per-token time impractical | 70B = single short "it runs" proof; smaller too-large model for the sweep (ADR-002) |
| Class mismatch on load | mandatory `AutoModel` (AF-1) |
| `C:` flooded by shards | configurable path off root + dry-run (AF-2, R5) |
| CUDA-only compression unavailable | FP16 baseline + GGUF/Ollama for quality axis (ADR-004) |
| Slow 9p I/O skews results | measure both `/mnt/c` and ext4 `~`; report as sensitivity (E1) |

## 8. Done when
Giant model emits coherent output via AirLLM `AutoModel` on CPU with shards on a non-system path, per-layer timeline captured, token never hard-coded — i.e., **KPI K1 met** and feeding the benchmark + theory sections.
