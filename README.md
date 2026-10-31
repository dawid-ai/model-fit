# model-fit

Tells you whether a local AI model will run well on a given computer, from the hardware alone:
how much memory it needs, how much of the machine that is, where it runs (GPU, partly offloaded,
or processor only), and roughly how many tokens per second it writes.

- Pure TypeScript: no I/O, no network, no dependencies. Safe to run in any process.
- Numbers only. Your app writes its own words for the user.
- The formulas and tables are ported from [llmfit](https://github.com/AlexsJones/llmfit) (MIT).
  The golden test vectors in `test/vectors/` come from real `llmfit --json plan` runs, so the
  results stay comparable.

## Install

Each version ships as a GitHub release asset, not on npm (0.2.0 is the last npm version).
Install the tarball by URL; your lockfile pins its hash:

```bash
pnpm add https://github.com/dawid-ai/model-fit/releases/download/v0.3.1/dawid-ai-model-fit-0.3.1.tgz
```

Every tarball carries a signed build attestation from the release workflow. Check it with:

```bash
gh attestation verify dawid-ai-model-fit-0.3.1.tgz -R dawid-ai/model-fit
```

## Use

```ts
import { fit, specFor } from "@dawid-ai/model-fit";

const profile = {
  ramBytes: 32 * 1024 ** 3,
  cpuCores: 16,
  gpus: [{ model: "AMD Radeon RX 7900 XTX", vramBytes: 24 * 1024 ** 3 }],
  bestVramBytes: 24 * 1024 ** 3,
  unifiedMemory: false,
};

const spec = specFor("qwen2.5:14b"); // a Hugging Face repo id or an Ollama tag
const result = fit(profile, spec ?? { params: 14 }, { ctx: 4096 });
// â†’ { needBytes, budgetBytes, share, runMode: "gpu", level: "perfect", tokPerSec: ~71, estimated }
```

`fit()` never guesses a pass: unknown memory gives `runMode: "none"` and `tokPerSec: null`, and
`estimated` says which parts came from fallbacks (no architecture data, an unknown GPU).

A model too big for the GPU runs partly offloaded (`runMode: "partial"`) only if the whole model
also fits in system RAM on its own, as in llmfit. VRAM and RAM are not added together: an 8 GB
card with 8 GB of RAM can't run a model that needs 10 GB.

Options let an app be stricter than llmfit: `unifiedUsable` (share of unified memory a model may
use, default 1), `minUsefulVramBytes` (a GPU below this counts as none, default 0), and
`reserveBytes` (memory held back from the GPU or unified pool for the desktop, default 0).
`kvBytesPerElement` sizes the KV cache (2 for fp16, 1 for q8_0) and `ddrGBps` sets system RAM
bandwidth for models partly in RAM (default 50).

### Mixture of experts

Pass `activeParams` (billions used per token), and `numExperts` / `activeExperts` when known.
Memory counts every expert; speed reads only the active ones, with llmfit's expert overhead and
memory-pressure factors. When all the experts don't fit the card but the active ones do,
`runMode` is `"moe-offload"`: the other experts stay in RAM.

### KV cache geometry

`layers`, `kvHeads` and `headDim` give the standard cache. For newer architectures add:

- `fullAttentionLayers`: layers that attend over the whole context. The others hold
  `slidingWindow` tokens (Gemma, gpt-oss) or no growing cache (linear or recurrent layers).
- `mlaKvWidth`: elements cached per token per layer for latent attention (DeepSeek, Kimi).
- `kvBytesPerToken: { full, sliding }`: a precomputed fp16 figure, when layers differ.

### Quality estimate

`qualityEstimate({ params, activeParams, family, generation, releaseMonth, nowMonth, quant })`
ports llmfit's quality score as a fallback for models without a benchmark: a continuous size curve,
a mixture of experts counted as âˆš(total Ã— active), family, generation and recency bonuses, and a
penalty per quantization. It isn't clamped at 100.

## Differences from llmfit

The numbers track llmfit closely (the golden vectors hold us within 10% on memory and 25% on
speed), but a few choices differ on purpose:

- **GPU first.** A model that fits the graphics card (up to 98% of it) runs there. llmfit 1.1.16
  may report a near-full card as partly offloaded to RAM, which halves its speed estimate.
- **Total RAM, not free RAM.** `fit()` uses the RAM you pass in. llmfit's live run uses the RAM
  free at that moment, which changes minute to minute; pass free RAM if that's what you want.
- **Fit levels** use llmfit's current source thresholds (60% / 85% / 98% of the memory pool, and
  "perfect" only on the GPU). The 1.1.16 binary uses older rules ("perfect" needs VRAM of at least
  1.2Ã— the need or need + 2 GB; processor-only caps at "marginal").
- **Whole-word GPU names.** The bandwidth table matches whole words, so an "RTX A1000" isn't read
  as an A100 or a "T400" as a T4. llmfit matches any substring.
- **Exact file sizes** (`sizeBytes`) are counted in real bytes (GiB), like device memory.
- **No speed for a model that won't fit.** llmfit still prints a speed for "too tight".
- **Small models.** Speed is memory bandwidth Ã· weights, as in llmfit, which overestimates very
  small models several times over. Apps should cap how they word big numbers.

## Data

The tables in `data/` are plain JSON so other languages can use them:

| File | What's in it |
|---|---|
| `data/models.json` | Architecture per model (params, layers, KV heads, head size), keyed by Hugging Face id, with Ollama tags |
| `data/gpu-bandwidth.json` | GPU memory bandwidth by name, ordered most-specific first |
| `data/quant.json` | Bytes per parameter and speed factors per quantization |
| `data/constants.json` | Overhead, efficiency, fit thresholds, fallback speed constants |

To add a model or a GPU, edit the JSON and run `pnpm test`. The data checks reject implausible
numbers and GPU names that an earlier, shorter entry would shadow.

## Develop

```bash
pnpm install
pnpm typecheck
pnpm test
```

## License

MIT. Includes llmfit's MIT notice for the ported formulas and tables. See `LICENSE`.
