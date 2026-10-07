import constants from "../data/constants.json" with { type: "json" };
import quantTable from "../data/quant.json" with { type: "json" };
import bandwidthTable from "../data/gpu-bandwidth.json" with { type: "json" };

// Hardware + a model → how much memory it needs, how much of the machine that is, and roughly how
// fast it will write. Pure: no I/O, no network, no dependencies. Numbers only — every app writes
// its own words. Formulas and tables are ported from llmfit 1.1.16 (MIT); see LICENSE.
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
  /** True when "GPU memory" is system RAM shared with the GPU (Apple Silicon). */
  unifiedMemory: boolean;
  /** Override the backend guess (unified → metal, NVIDIA → cuda, anything else → vulkan). */
  backend?: Backend;
}

export type ModelKind = "text" | "stt" | "ocr" | "vision" | "embedding" | "image";

export interface FitModel {
  /** Billions of parameters. */
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
}

export interface FitOptions {
  /** Context window in tokens (default 4096). */
  ctx?: number;
  /** Share of unified memory a model may use (default 1, llmfit's assumption). */
  unifiedUsable?: number;
  /** A GPU with less dedicated VRAM than this counts as no GPU (default 0). */
  minUsefulVramBytes?: number;
}

export type RunMode = "gpu" | "partial" | "cpu" | "none";
export type FitLevel = "perfect" | "good" | "marginal" | "too-tight";

export interface FitResult {
  /** Memory the model needs, in bytes (weights + KV cache + runtime overhead). */
  needBytes: number;
  /** The memory pool it would run from (VRAM, unified memory, or RAM), 0 when unknown. */
  budgetBytes: number;
  /** needBytes / budgetBytes; Infinity when the pool is unknown or empty. */
  share: number;
  /** Where it runs: all on the GPU, partly offloaded to RAM, on the processor, or not at all. */
  runMode: RunMode;
  level: FitLevel;
  /** Estimated tokens per second; null when it won't run, memory is unknown, or it isn't a text model. */
  tokPerSec: number | null;
  /** Which parts are estimates rather than read from the model's own data. */
  estimated: { weights: boolean; kv: boolean; speed: boolean };
}

type QuantRow = { bpp: number; speedBpp: number; speedMultiplier: number };
const QUANTS = quantTable.quants as Record<string, QuantRow>;
function quantRow(q: string | undefined): QuantRow {
  const key = q ?? constants.defaultQuant;
  return Object.hasOwn(QUANTS, key) ? QUANTS[key]! : quantTable.default;
}

/** Memory bandwidth (GB/s) for a GPU name, or null when unknown or a mobile part. A table entry
 *  matches only as a whole word (llmfit matches any substring, which reads an "RTX A1000" as an
 *  A100 and a "T400" as a T4). */
export function gpuBandwidthGBps(name: string): number | null {
  const lower = name.toLowerCase();
  if (bandwidthTable.mobileMarkers.some((m) => lower.includes(m))) return null;
  return bandwidthTable.gpus.find((g) => wholeWord(lower, g.match))?.gbps ?? null;
}

function wholeWord(haystack: string, needle: string): boolean {
  for (let i = haystack.indexOf(needle); i !== -1; i = haystack.indexOf(needle, i + 1)) {
    const before = haystack[i - 1];
    const after = haystack[i + needle.length];
    if ((before === undefined || !/[a-z0-9]/.test(before)) && (after === undefined || !/[a-z0-9]/.test(after))) return true;
  }
  return false;
}

function backendOf(p: FitProfile): Backend {
  if (p.backend) return p.backend;
  if (p.unifiedMemory) return "metal";
  return p.gpus.some((g) => /nvidia|geforce|rtx|gtx|quadro|tesla/i.test(`${g.vendor ?? ""} ${g.model}`)) ? "cuda" : "vulkan";
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

  // An exact file size is real bytes, compared against device memory in GiB — so GiB here too.
  const weightsGB = model.sizeBytes !== undefined ? model.sizeBytes / GiB : params * q.bpp;
  const exactKv = model.layers !== undefined && model.kvHeads !== undefined && model.headDim !== undefined;
  const kvGB = exactKv
    ? (2 * model.layers! * model.kvHeads! * model.headDim! * ctx * constants.kvBytesPerElement) / GiB
    : constants.kvFallbackGBPerParamBPerToken * params * ctx;
  const needGB = weightsGB + kvGB + constants.overheadGB;

  const ramGB = profile.ramBytes !== null && profile.ramBytes > 0 ? profile.ramBytes / GiB : 0;
  const vramBytes = profile.bestVramBytes ?? 0;
  const hasGpu = !profile.unifiedMemory && vramBytes > 0 && vramBytes >= (opts.minUsefulVramBytes ?? 0);
  const max = constants.levels.marginalMax;

  let runMode: RunMode = "none";
  let poolGB = 0;
  if (profile.unifiedMemory) {
    poolGB = ramGB * (opts.unifiedUsable ?? 1);
    if (poolGB > 0 && needGB / poolGB <= max) runMode = "gpu";
  } else if (hasGpu) {
    poolGB = vramBytes / GiB;
    if (needGB / poolGB <= max) runMode = "gpu";
    else if (ramGB > 0 && needGB / ramGB <= max) {
      runMode = "partial"; // the RAM pool carries it; the GPU takes what fits
      poolGB = ramGB;
    }
  } else {
    poolGB = ramGB;
    if (poolGB > 0 && needGB / poolGB <= max) runMode = "cpu";
  }
  const share = poolGB > 0 ? needGB / poolGB : Number.POSITIVE_INFINITY;

  const tok = (model.kind ?? "text") === "text" && runMode !== "none" ? speed(profile, params, q, runMode) : null;
  return {
    needBytes: needGB * GiB,
    budgetBytes: poolGB * GiB,
    share,
    runMode,
    level: levelOf(share, runMode),
    tokPerSec: tok?.value ?? null,
    estimated: { weights: model.sizeBytes === undefined, kv: !exactKv, speed: tok?.estimatedConstant ?? true },
  };
}

function speed(profile: FitProfile, params: number, q: QuantRow, runMode: RunMode): { value: number; estimatedConstant: boolean } {
  const factor = constants.runModeFactor[runMode as "gpu" | "partial" | "cpu"];
  const gpuName = profile.gpus.find((g) => g.vramBytes === profile.bestVramBytes)?.model ?? profile.gpus[0]?.model;
  const bw = runMode !== "cpu" && gpuName ? gpuBandwidthGBps(gpuName) : null;
  if (bw !== null) {
    // Generation is memory-bound: each token reads the weights once.
    return { value: Math.max(0.1, (bw / (params * q.speedBpp)) * constants.efficiency * factor), estimatedConstant: false };
  }
  // Unknown GPU (or CPU-only): a fixed throughput constant per backend.
  const backend = runMode === "cpu" ? (backendOf(profile) === "cpu-arm" ? "cpu-arm" : "cpu") : backendOf(profile);
  let v = (constants.fallbackK[backend] / params) * q.speedMultiplier;
  if ((profile.cpuCores ?? 0) >= constants.manyCoresThreshold) v *= constants.manyCoresBonus;
  return { value: Math.max(0.1, v * factor), estimatedConstant: true };
}
