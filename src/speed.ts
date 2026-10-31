import constants from "../data/constants.json" with { type: "json" };
import bandwidthTable from "../data/gpu-bandwidth.json" with { type: "json" };
import type { Backend, FitModel, FitProfile, RunMode } from "./fit.js";

// Words-per-second estimates (decode only), ported from llmfit `estimate_tps` (fit.rs). Generation
// is memory-bound: each token reads the weights it uses once, so speed is bandwidth ÷ bytes read.

export type QuantRow = { bpp: number; speedBpp: number; speedMultiplier: number; qualityPenalty: number };

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

export function backendOf(p: FitProfile): Backend {
  if (p.backend) return p.backend;
  if (p.unifiedMemory) return "metal";
  return p.gpus.some((g) => /nvidia|geforce|rtx|gtx|quadro|tesla/i.test(`${g.vendor ?? ""} ${g.model}`)) ? "cuda" : "vulkan";
}

/** The billions of parameters read per token: the active ones for a mixture-of-experts model. */
export function paramsPerToken(model: FitModel): number {
  return Math.max(model.activeParams ?? model.params, 0.1);
}

function moeOverhead(model: FitModel): number {
  if (model.numExperts === undefined) return constants.moe.overheadUnknown;
  return constants.moe.overheadByExperts.find((r) => model.numExperts! <= r.maxExperts)!.overhead;
}

// llmfit fit.rs:1959: when the experts crowd the card, more of them miss its fastest memory.
function moePressure(model: FitModel, q: QuantRow, poolGB: number): number {
  if (poolGB <= 0 || model.numExperts === undefined || model.activeExperts === undefined) return 1;
  const util = (model.params * q.bpp) / poolGB;
  if (util < constants.moe.pressureStart || util > 1) return 1;
  return Math.max(1 - (util - constants.moe.pressureStart) * (1 - model.activeExperts / model.numExperts), constants.moe.pressureFloor);
}

export interface SpeedInput {
  profile: FitProfile;
  model: FitModel;
  q: QuantRow;
  runMode: Exclude<RunMode, "none">;
  /** The GPU's memory pool in GB (VRAM, or the usable share of unified memory). */
  gpuPoolGB: number;
  /** GB of the GPU pool left for weights after the KV cache and overhead (partial offload). */
  gpuWeightsGB: number;
  ddrGBps: number;
}

export function estimateSpeed(s: SpeedInput): { value: number; estimatedConstant: boolean } {
  const { profile, model, q, runMode } = s;
  const moe = model.activeParams !== undefined && model.activeParams < model.params;
  const params = paramsPerToken(model);
  const gpuName = profile.gpus.find((g) => g.vramBytes === profile.bestVramBytes)?.model ?? profile.gpus[0]?.model;
  const tableBw = runMode !== "cpu" && gpuName ? gpuBandwidthGBps(gpuName) : null;
  const cores = (profile.cpuCores ?? 0) >= constants.manyCoresThreshold ? constants.manyCoresBonus : 1;
  const backend = runMode === "cpu" ? (backendOf(profile) === "cpu-arm" ? "cpu-arm" : "cpu") : backendOf(profile);
  // An unknown GPU gets a bandwidth implied by its backend's fallback constant, so every path below
  // reads the same way (llmfit does this for MoE offload, fit.rs:2061).
  const bw = tableBw ?? (constants.fallbackK[backend] * q.speedMultiplier * q.speedBpp * cores) / constants.efficiency;
  const estimatedConstant = tableBw === null;
  const eff = constants.efficiency;

  let value: number;
  if (runMode === "cpu") {
    value = (constants.fallbackK[backend] / params) * q.speedMultiplier * cores * constants.runModeFactor.cpu;
  } else if (runMode === "moe-offload") {
    // Active weights stream from RAM and through the GPU every token (llmfit fit.rs:1899).
    const activeGB = params * q.speedBpp;
    value = (1 / (activeGB / s.ddrGBps + activeGB / (bw * eff))) * constants.runModeFactor.moeOffload;
  } else if (runMode === "partial" && tableBw === null) {
    // Unknown GPU: llmfit keeps the flat fallback constant × the partial-offload factor.
    value = (constants.fallbackK[backend] / params) * q.speedMultiplier * cores * constants.runModeFactor.partial;
  } else if (runMode === "partial") {
    // Dense layers split between GPU and RAM (llmfit 1.1.18 dense_cpu_offload_tps): the GPU holds
    // as many whole layers of the per-token weights as fit beside the KV cache and overhead, and
    // the rest streams from system memory. Replaces 1.1.16's flat half speed.
    const weightsGB = params * q.speedBpp;
    const room = Math.max(s.gpuWeightsGB, 0);
    let gpuGB = Math.min(weightsGB, room);
    if (model.layers !== undefined && model.layers > 0) {
      const slice = weightsGB / model.layers;
      gpuGB = Math.min(Math.floor(room / slice), model.layers) * slice;
    }
    const ramGB = weightsGB - gpuGB;
    value = ramGB <= 0 ? (bw / weightsGB) * eff : 1 / (gpuGB / (bw * eff) + ramGB / s.ddrGBps);
  } else if (moe) {
    value = (bw / (params * q.bpp)) * eff * moeOverhead(model) * moePressure(model, q, s.gpuPoolGB);
  } else {
    value = (bw / (params * q.speedBpp)) * eff;
  }
  return { value: Math.max(0.1, value), estimatedConstant };
}
