// Regenerate test/vectors/ by running the real llmfit `plan` against each model on several hardware
// setups. Usage: node scripts/gen-vectors.mjs <path-to-llmfit-executable>
// Get llmfit from https://github.com/AlexsJones/llmfit/releases (verify the .sha256). The real-box
// setup describes Dawid's AMD Radeon RX 7900 XTX machine — re-run that one on that machine, or edit it.
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const exe = process.argv[2];
if (!exe) throw new Error("usage: node scripts/gen-vectors.mjs <path-to-llmfit>");
const out = path.join(here, "..", "test", "vectors");
mkdirSync(out, { recursive: true });
const GiB = 1024 ** 3;
const today = new Date().toISOString().slice(0, 10);

// What llmfit knows about each model: params from its DB; layers/kvHeads/headDim only where its DB
// has them (the gated Llama/Gemma entries don't, so llmfit uses its params×ctx KV fallback there).
const MODELS = [
  { id: "qwen2.5:0.5b", hf: "Qwen/Qwen2.5-0.5B-Instruct", params: 0.494032768, layers: 24, kvHeads: 2, headDim: 64 },
  { id: "qwen2.5:1.5b", hf: "Qwen/Qwen2.5-1.5B-Instruct", params: 1.543714304, layers: 28, kvHeads: 2, headDim: 128 },
  { id: "qwen2.5:3b", hf: "Qwen/Qwen2.5-3B-Instruct", params: 3.085938688, layers: 36, kvHeads: 2, headDim: 128 },
  { id: "qwen2.5:7b", hf: "Qwen/Qwen2.5-7B-Instruct", params: 7.615616512, layers: 28, kvHeads: 4, headDim: 128 },
  { id: "qwen2.5:14b", hf: "Qwen/Qwen2.5-14B-Instruct", params: 14.770033664, layers: 48, kvHeads: 8, headDim: 128 },
  { id: "qwen2.5:32b", hf: "Qwen/Qwen2.5-32B-Instruct", params: 32.763876352, layers: 64, kvHeads: 8, headDim: 128 },
  { id: "llama3.2:1b", hf: "meta-llama/Llama-3.2-1B-Instruct", params: 1.2358144, layers: 16, kvHeads: 8, headDim: 64 },
  { id: "llama3.2:3b", hf: "meta-llama/Llama-3.2-3B-Instruct", params: 3.212749824, layers: 28, kvHeads: 8, headDim: 128 },
  { id: "gemma2:27b", hf: "google/gemma-2-27b-it", params: 27.22712832, layers: 46, kvHeads: 16, headDim: 128 },
];

const SETUPS = [
  {
    name: "24gb-7900xtx-real",
    source: `llmfit 1.1.16 plan --json on Dawid's real box (AMD Radeon RX 7900 XTX 24 GB, 95.7 GB RAM, 24 cores), ${today}`,
    args: [],
    profile: { ramBytes: 95.7 * GiB, cpuCores: 24, gpus: [{ model: "AMD Radeon RX 7900 XTX", vramBytes: 24 * GiB, vendor: "AMD" }], bestVramBytes: 24 * GiB, unifiedMemory: false, backend: "vulkan" },
  },
  {
    name: "8gb-named-card",
    source: `llmfit 1.1.16 plan --json --memory 8G --ram 16G --cpu-cores 8 (card name kept: AMD Radeon RX 7900 XTX → 960 GB/s), ${today}`,
    args: ["--memory", "8G", "--ram", "16G", "--cpu-cores", "8"],
    profile: { ramBytes: 16 * GiB, cpuCores: 8, gpus: [{ model: "AMD Radeon RX 7900 XTX", vramBytes: 8 * GiB, vendor: "AMD" }], bestVramBytes: 8 * GiB, unifiedMemory: false, backend: "vulkan" },
  },
  {
    name: "16gb-apple-m2",
    source: `llmfit 1.1.16 plan --json --profile apple-m2-16gb (custom profile: 16 GB unified, 100 GB/s), ${today}`,
    args: ["--profile", path.join(here, "llmfit-profiles", "apple-m2-16gb.json")],
    profile: { ramBytes: 16 * GiB, cpuCores: 8, gpus: [{ model: "Apple M2", vramBytes: null, vendor: "Apple" }], bestVramBytes: null, unifiedMemory: true, backend: "metal" },
  },
  {
    name: "16gb-cpu-only",
    source: `llmfit 1.1.16 plan --json --memory 0G --ram 16G --cpu-cores 8, ${today}`,
    args: ["--memory", "0G", "--ram", "16G", "--cpu-cores", "8"],
    profile: { ramBytes: 16 * GiB, cpuCores: 8, gpus: [], bestVramBytes: null, unifiedMemory: false, backend: "cpu" },
  },
];

const RUN_MODE = { Gpu: "gpu", CpuOffload: "partial", CpuOnly: "cpu" };

for (const s of SETUPS) {
  const vectors = [];
  for (const m of MODELS) {
    let plan;
    try {
      plan = JSON.parse(execFileSync(exe, [...s.args, "plan", m.hf, "--context", "4096", "--quant", "Q4_K_M", "--json"], { encoding: "utf8" }));
    } catch (e) {
      console.error("FAILED", s.name, m.hf, String(e.stdout ?? e.message).slice(0, 300));
      continue;
    }
    const needGB = plan.run_paths[0].minimum.vram_gb ?? plan.run_paths[0].minimum.ram_gb;
    const cur = plan.current ?? {};
    // llmfit still reports a run mode + speed for a TooTight model; for us that model won't run.
    const runMode = cur.fit_level === "TooTight" ? "none" : RUN_MODE[cur.run_mode] ?? "none";
    const model = { id: m.id, params: m.params, quant: "Q4_K_M", kind: "text" };
    if (m.layers) Object.assign(model, { layers: m.layers, kvHeads: m.kvHeads, headDim: m.headDim });
    vectors.push({
      source: s.source,
      input: { profile: s.profile, model, opts: { ctx: 4096 } },
      expect: { needGB, tokPerSec: runMode === "none" ? null : cur.estimated_tps ?? null, runMode, llmfitFit: cur.fit_level ?? null },
    });
    console.log(s.name.padEnd(20), m.id.padEnd(14), needGB.toFixed(3), runMode.padEnd(8), (cur.estimated_tps ?? 0).toFixed(2), cur.fit_level);
  }
  writeFileSync(path.join(out, `${s.name}.json`), JSON.stringify(vectors, null, 2) + "\n");
}

// ── Hand-derived vectors (llmfit can't produce these directly) ────────────────────────────────
// Re-implements llmfit 1.1.16's formulas independently of the module (fit.rs estimate_tps,
// models.rs estimate_memory_gb_with_kv) for the two inputs the CLI can't run: the 0.5B (its model
// selector is ambiguous in llmfit's DB) and a GPU whose name isn't in the bandwidth table.
function llmfitNeedGB(m) {
  return m.params * 0.58 + (2 * m.layers * m.kvHeads * m.headDim * 4096 * 2) / GiB + 0.5;
}
const half = MODELS[0]; // qwen2.5:0.5b
const handSource = `hand-derived from llmfit 1.1.16 formulas (fit.rs estimate_tps; models.rs estimate_memory_gb_with_kv), ${today}`;
const handModel = (m) => ({ id: m.id, params: m.params, quant: "Q4_K_M", kind: "text", layers: m.layers, kvHeads: m.kvHeads, headDim: m.headDim });
const hand = [];
// 0.5B on each real-run setup.
const BW = { "24gb-7900xtx-real": 960, "8gb-named-card": 960, "16gb-apple-m2": 100 };
for (const s of SETUPS) {
  const need = llmfitNeedGB(half);
  const tok = BW[s.name] !== undefined
    ? (BW[s.name] / (half.params * 0.5)) * 0.55 * 1.0
    : (70 / half.params) * 1.15 * 1.1 * 0.3; // CPU-only: cpu K, quant multiplier, 8+ cores bonus, cpu factor
  hand.push({ source: `${handSource}; setup ${s.name}`, input: { profile: s.profile, model: handModel(half), opts: { ctx: 4096 } }, expect: { needGB: need, tokPerSec: tok, runMode: s.name === "16gb-cpu-only" ? "cpu" : "gpu" } });
}
// Unknown 8 GB card (name not in the table) → fixed Vulkan constant: K / params × quant × cores bonus.
const unknown = { ramBytes: 16 * GiB, cpuCores: 8, gpus: [{ model: "Generic Graphics 8GB", vramBytes: 8 * GiB }], bestVramBytes: 8 * GiB, unifiedMemory: false, backend: "vulkan" };
for (const m of [MODELS[2], MODELS[3], MODELS[4]]) {
  const need = llmfitNeedGB(m);
  const gpu = need / 8 <= 0.98;
  const tok = (150 / m.params) * 1.15 * 1.1 * (gpu ? 1.0 : 0.5);
  hand.push({ source: `${handSource}; unknown GPU, vulkan K=150`, input: { profile: unknown, model: handModel(m), opts: { ctx: 4096 } }, expect: { needGB: need, tokPerSec: tok, runMode: gpu ? "gpu" : "partial" } });
}
writeFileSync(path.join(out, "hand-derived.json"), JSON.stringify(hand, null, 2) + "\n");
console.log("hand-derived", hand.length);
