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

```bash
npm install @dawid-ai/model-fit
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
// → { needBytes, budgetBytes, share, runMode: "gpu", level: "perfect", tokPerSec: ~71, estimated }
```

`fit()` never guesses a pass: unknown memory gives `runMode: "none"` and `tokPerSec: null`, and
`estimated` says which parts came from fallbacks (no architecture data, an unknown GPU).

Options let an app be stricter than llmfit: `unifiedUsable` (share of Apple unified memory a model
may use, default 1) and `minUsefulVramBytes` (a GPU below this counts as none, default 0).

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
