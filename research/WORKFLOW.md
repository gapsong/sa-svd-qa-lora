# Research workflow playbook

Distilled from the 2026-06 SA-SVD red-team arc (red-team.md, journal v1-v13).
Every rule below was paid for with a real mistake in this repo; the mistake
is named so you learn the rule, not just follow it. The two project skills
(/research-cycle, /research-interpret) operationalize this file.

## The loop, one cycle

```
0. Claim inventory      what exactly do we believe? verbatim, with source
1. Red-team             rank attacks on the claim by severity, cheapest decisive first
2. Register prediction  write down what each arm WILL show, before running
3. Discrimination check validate any proxy against a known ordering first
4. Noise calibration    rerun one identical cycle; threshold = ~2x observed noise
5. Guarded run          VRAM guard, background + monitor, never block the chat
6. Interpret            anomaly checklist before believing any number
7. Journal              append-only; failures are data; takeaway in one line
8. Verdict              KEEP / REVERT / FAILED against the calibrated threshold
```

## Step rules and the mistakes that bought them

**0/1. Attack the claim before extending it.** The headline "only the init
differs" was strictly false (the residual swap changes the quantized base
too). One free CPU probe (E1) settled what could have been argued about
for weeks. Rank attacks by cost-to-kill, not by how interesting they are.

**2. Registered predictions.** Write the expected outcome of every arm
BEFORE launching. Twice the prediction was wrong (E2: confused quantization
error with function; E6: spectrum did matter) and the written prediction is
what made the surprise undeniable instead of quietly rationalized.

**3. Discrimination check.** Before trusting a cheap proxy (smaller model,
shorter budget, eval subset), verify it reproduces ONE ordering you already
trust. The 135M proxy inverted the known sa_svd-vs-zero ordering (journal
v1, exp 0c); without the check, weeks of loop cycles would have optimized
the wrong sign. All three shortcuts failed validation here.

**4. Calibrate noise before comparing.** Two bit-identical reference cycles
differed by 2.30 PPL (GPU nondeterminism). Single-seed deltas smaller than
that are weather, not signal. E3 lesson: the baseline's seed spread (4.2)
was two thirds of the headline gap.

**5. Budget the GPU, never block.** Keep ~8 GB reserved for other apps:
every run goes through research/gpu_guard.sh, batch sized so run + 8 GiB
fits in 24 (1.7B at batch 2 x accum 8 is ~13 GiB). Launch via background
task + Monitor with failure-covering filters (START/DONE/FAILED lines, not
just success greps). Waiting in the foreground wastes the session.

**6. Interpret with a checklist, not vibes** (this is /research-interpret):
- step-10 loss vs known anchors (SmolLM2 1.7B: healthy ~7-10, damaged 13+)
- grad_norm: finite band ~9-17 here; ANY inf step is a finding, not noise
- final PPL anchors: ~12.2k = unadapted 2-bit model (adapter learned
  nothing), ~36 = sa_svd, ~34 = written-Kaiming, ~30 = usable threshold
- seed spread vs the calibrated noise; unseeded variance is a red flag
  that something upstream is non-deterministic (v11: uninitialized memory)
- time per phase: GPTQ ~2 min, train ~0.8 s/step, eval ~2 min at 1.7B on a
  free 4090; 2-3x slower means GPU contention, check nvidia-smi before
  concluding anything about the method
- "did not train" is never a regime property until the optimizer state and
  the INIT TENSORS are probed directly. The Qwen2 "inf gradients regime"
  was uninitialized memory from an upstream bug (v11-v13).

**7/8. Journal protocol.** Append-only, one entry per cycle: id, date,
hypothesis (one line), change (one line), per-seed metrics, mean, verdict,
takeaway. Never edit old entries; correct them forward (v13 corrects v11).
Record FAILED cycles with the same care; they carry most of the learning.

**Toolchain distrust rule (v11-v13).** When a baseline behaves impossibly
(unseeded spread, exact zeros, saddle points), stop theorizing about the
method and probe the tensors and the library source. Both bugs found here
(gptqmodel init leak, PEFT variant re-init) were upstream, unreported, and
invisible at the API surface. Budget one probe script before any
mechanism story that blames your own method.

## Reading list (the learning track)

Each maps to a decision made in this repo:
- LoRA (Hu et al. 2021): why B=0 at init makes the adapter a no-op, and
  why A=B=0 is a saddle (grad A ~ B, grad B ~ A).
- QLoRA (Dettmers et al. 2023): quantize-then-adapt baseline; why the
  adapter must compensate quantization damage.
- QA-LoRA (Xu et al. 2023): the group-constant constraint that makes the
  adapter mergeable into zero-points; what qalora_group_size means.
- PiSSA (Meng et al. 2024): principal-component init; SA-SVD is its
  quantization-group-aware variant.
- LoRA+ (Hayou et al. 2024): asymmetric A/B learning rates; the reason
  factor-asymmetric inits are on the idea list (program.md idea 2).
- karpathy/autoresearch: one mutable file, one metric, fixed budget,
  keep-or-revert; the template behind autoresearch/.

## Current state pointers

- Corrected baseline: written-Kaiming, --method kaiming in reproduce.py.
- Open: TinyLlama/Qwen2 kaiming reruns (research/run_kaiming_reruns.sh),
  upstream issue filing (research/upstream-bug-reports.md, needs review),
  fork test (qzero_unquantized), then ONE coherent public-text rewrite.
