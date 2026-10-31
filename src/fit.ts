import constants from "../data/constants.json" with { type: "json" };
import quantTable from "../data/quant.json" with { type: "json" };
import { kvCacheGB } from "./kv.js";
import { estimateSpeed, paramsPerToken, type QuantRow } from "./speed.js";

// Hardware + a model → how much memory it needs, how much of the machine that is, and roughly how
// fast it will write. Pure: no I/O, no network, no dependencies. Formulas and tables are ported from
// llmfit 1.1.18 (MIT); see LICENSE.
//
// ponytail: memory mirrors llmfit's units for parity — weights in decimal GB (params_b × bytes per
// param), KV cache and device memory in GiB. Unify the units if parity with llmfit stops mattering.

const GiB = 1024 ** 3;

/** The GPU backend, which picks the fallback speed constant when a GPU isn't in the bandwidth table. */
export type Backend = "cuda" | "rocm" | "vulkan" | "metal" | "sycl" | "cpu" | "cpu-arm";

/** What the machine has. Every field may be unknown (null) — the result says so instead of guessing. */
export interface FitProfile {
  ramBytes: number | null;
  cpuCores: number | null;
  gpus: { model: string; vramBytes: number | null; vendor?: string }[];
  /** Largest dedicated VRAM, or null when there's no readable dedicated VRAM. */
  bestVramBytes: number | null;
  /** True when "GPU memory" is system RAM shared with the GPU (Apple Silicon, Ryzen AI Max, GB10). */
  unifiedMemory: boolean;
  /** Override the backend guess (unified → metal, NVIDIA → cuda, anything else → vulkan). */
  backend?: Backend;
}

export type ModelKind = "text" | "stt" | "ocr" | "vision" | "embedding" | "image";

export interface FitModel {
  /** Billions of parameters (all of them, for a mixture-of-experts model). */
  params: number;
  /** Quantization label (e.g. "Q4_K_M"); defaults to Q4_K_M. */
  quant?: string;
  /** Exact weights file size in bytes — when known it replaces the params × bytes-per-param estimate. */
  sizeBytes?: number;
  layers?: number;
  kvHeads?: number;
  headDim?: number;
  /** Non-text models get a memory verdict but no words-per-second (defaults to "text"). */
  kind?: ModelKind;
  /** Mixture of experts: billions of parameters used per token. Memory still counts `params`. */
  activeParams?: number;
  numExperts?: number;
  activeExperts?: number;
  /** Layers that attend over the whole context. The rest keep `slidingWindow` tokens, or no KV
   *  cache at all (linear or recurrent layers). Defaults to every layer. */
  fullAttentionLayers?: number;
  slidingWindow?: number;
  /** Multi-head latent attention (DeepSeek, Kimi): cached elements per token per layer. */
  mlaKvWidth?: number;
  /** Precomputed KV cache at fp16, summed over layers: bytes per token for the full-context
   *  layers, and for the sliding-window layers (which stop growing at `slidingWindow` tokens).
   *  Wins over the per-layer fields; use it when layers differ (per-layer heads, shared KV). */
  kvBytesPerToken?: { full: number; sliding?: number };
}

export interface FitOptions {
  /** Context window in tokens (default 4096). */
  ctx?: number;
  /** Share of unified memory a model may use (default 1, llmfit's assumption). */
  unifiedUsable?: number;
  /** A GPU with less dedicated VRAM than this counts as no GPU (default 0). */
  minUsefulVramBytes?: number;
  /** Memory held back from the GPU or unified pool for the desktop and driver (default 0). */
  reserveBytes?: number;
  /** Bytes per KV-cache element: 2 for fp16 (default), 1 for q8_0. */
  kvBytesPerElement?: number;
  /** System RAM bandwidth in GB/s for models partly in RAM (default 50, llmfit's fallback). */
  ddrGBps?: number;
}

export type RunMode = "gpu" | "moe-offload" | "partial" | "cpu" | "none";
export type FitLevel = "perfect" | "good" | "marginal" | "too-tight";

export interface FitResult {
  /** Memory the model needs, in bytes (weights + KV cache + runtime overhead). */
  needBytes: number;
  /** The memory pool it would run from (VRAM, unified memory, or RAM), 0 when unknown. */
  budgetBytes: number;
  /** needBytes / budgetBytes (GPU-side need for MoE offload); Infinity when the pool is unknown. */
  share: number;
  /** Where it runs: all on the GPU, MoE experts in RAM, partly offloaded, on the processor, or not at all. */
  runMode: RunMode;
  level: FitLevel;
  /** Estimated tokens per second; null when it won't run, memory is unknown, or it isn't a text model. */
  tokPerSec: number | null;
  /** The KV cache part of `needBytes`. */
  kvBytes: number;
  /** Which parts are estimates rather than read from the model's own data. */
  estimated: { weights: boolean; kv: boolean; speed: boolean };
}

const QUANTS = quantTable.quants as Record<string, QuantRow>;
export function quantRow(q: string | undefined): QuantRow {
  const key = (q ?? constants.defaultQuant).toUpperCase();
  return Object.hasOwn(QUANTS, key) ? QUANTS[key]! : (quantTable.default as QuantRow);
}

function levelOf(share: number, runMode: RunMode): FitLevel {
  const { perfectMax, goodMax, marginalMax } = constants.levels;
  let level: FitLevel =
    !Number.isFinite(share) || share > marginalMax ? "too-tight" : share <= perfectMax ? "perfect" : share <= goodMax ? "good" : "marginal";
  // "Perfect" means room to spare AND all on the GPU; offload and CPU paths cap at "good".
  if (level === "perfect" && runMode !== "gpu") level = "good";
  return level;
}

/** Score one model against one machine. */
export function fit(profile: FitProfile, model: FitModel, opts: FitOptions = {}): FitResult {
  const ctx = opts.ctx ?? constants.defaultContext;
  const q = quantRow(model.quant);
  const params = Math.max(model.params, 0.1);
  const reserveGB = Math.max(opts.reserveBytes ?? 0, 0) / GiB;
  const ddr = opts.ddrGBps ?? constants.ddrGBps;

  // An exact file size is real bytes, compared against device memory in GiB — so GiB here too.
  const weightsGB = model.sizeBytes !== undefined ? model.sizeBytes / GiB : params * q.bpp;
  const kv = kvCacheGB(model, ctx, opts.kvBytesPerElement);
  const needGB = weightsGB + kv.gb + constants.overheadGB;

  const ramGB = profile.ramBytes !== null && profile.ramBytes > 0 ? profile.ramBytes / GiB : 0;
  const vramBytes = profile.bestVramBytes ?? 0;
  const hasGpu = !profile.unifiedMemory && vramBytes > 0 && vramBytes >= (opts.minUsefulVramBytes ?? 0);
  const max = constants.levels.marginalMax;
  const moe = model.activeParams !== undefined && model.activeParams < params;

  let runMode: RunMode = "none";
  let poolGB = 0;
  let gpuPoolGB = 0;
  let shareNeedGB = needGB;
  if (profile.unifiedMemory) {
    poolGB = gpuPoolGB = Math.max(ramGB * (opts.unifiedUsable ?? 1) - reserveGB, 0);
    if (poolGB > 0 && needGB / poolGB <= max) runMode = "gpu";
  } else if (hasGpu) {
    poolGB = gpuPoolGB = Math.max(vramBytes / GiB - reserveGB, 0);
    // Active weights on the GPU, the other experts in RAM (llmfit moe_offload_path, plus the KV cache).
    const activeGpuGB = paramsPerToken(model) * q.bpp * constants.moe.activeVramFactor + kv.gb + constants.overheadGB;
    const expertsRamGB = Math.max(weightsGB - paramsPerToken(model) * q.bpp, 0);
    if (poolGB > 0 && needGB / poolGB <= max) runMode = "gpu";
    else if (moe && poolGB > 0 && activeGpuGB / poolGB <= max && ramGB > 0 && expertsRamGB / ramGB <= max) {
      runMode = "moe-offload";
      shareNeedGB = activeGpuGB;
    } else if (ramGB > 0 && needGB / ramGB <= max) {
      runMode = "partial"; // the RAM pool carries it; the GPU takes what fits
      poolGB = ramGB;
    }
  } else {
    poolGB = ramGB;
    if (poolGB > 0 && needGB / poolGB <= max) runMode = "cpu";
  }
  const share = poolGB > 0 ? shareNeedGB / poolGB : Number.POSITIVE_INFINITY;

  const tok =
    (model.kind ?? "text") === "text" && runMode !== "none"
      ? estimateSpeed({ profile, model, q, runMode, gpuPoolGB, gpuWeightsGB: gpuPoolGB - kv.gb - constants.overheadGB, ddrGBps: ddr })
      : null;
  return {
    needBytes: needGB * GiB,
    budgetBytes: poolGB * GiB,
    share,
    runMode,
    level: levelOf(share, runMode),
    tokPerSec: tok?.value ?? null,
    kvBytes: kv.gb * GiB,
    estimated: { weights: model.sizeBytes === undefined, kv: !kv.exact, speed: tok?.estimatedConstant ?? true },
  };
}
