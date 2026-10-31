import { describe, expect, it } from "vitest";
import { effectiveParams, fit, kvCacheGB, qualityEstimate, quantRow, type FitModel, type FitProfile } from "../src/index.js";

// 0.3.0: the parts of llmfit 1.1.18 that 0.2.0 left out (mixture of experts, richer KV geometry,
// layer-split offload speed, a quality estimate), plus a memory reserve.

const GiB = 1024 ** 3;
const card = (vramGiB: number, ramGiB = 64, name = "NVIDIA GeForce RTX 4090"): FitProfile => ({
  ramBytes: ramGiB * GiB,
  cpuCores: 16,
  gpus: [{ model: name, vramBytes: vramGiB * GiB }],
  bestVramBytes: vramGiB * GiB,
  unifiedMemory: false,
});

// gpt-oss-20b-like and Qwen3-30B-A3B-like geometries.
const moe20: FitModel = { params: 20.9, activeParams: 3.6, numExperts: 32, activeExperts: 4, quant: "MXFP4", layers: 24, kvHeads: 8, headDim: 64 };
const moe30: FitModel = { params: 30.5, activeParams: 3.3, numExperts: 128, activeExperts: 8, layers: 48, kvHeads: 4, headDim: 128 };
const dense30: FitModel = { params: 30.5, layers: 48, kvHeads: 4, headDim: 128 };

describe("mixture of experts", () => {
  it("memory counts every expert, speed only the active ones", () => {
    const m = fit(card(24), moe30);
    const d = fit(card(24), dense30);
    expect(m.needBytes).toBeCloseTo(d.needBytes, 0);
    expect(m.runMode).toBe("gpu");
    // llmfit's overhead for 128 experts (0.4) and memory pressure keep it at ~2.8×, not 9×.
    expect(m.tokPerSec!).toBeGreaterThan(d.tokPerSec! * 2);
  });

  it("a card too small for all the experts keeps the active ones and streams the rest from RAM", () => {
    const r = fit(card(8, 32), moe30);
    expect(r.runMode).toBe("moe-offload");
    expect(r.level).not.toBe("perfect");
    // Faster than the same size dense model split between card and RAM.
    const dense = fit(card(8, 32), dense30);
    expect(dense.runMode).toBe("partial");
    expect(r.tokPerSec!).toBeGreaterThan(dense.tokPerSec!);
  });

  it("a dense model never takes the expert path", () => {
    expect(fit(card(8, 32), dense30).runMode).not.toBe("moe-offload");
  });

  it("MXFP4 (gpt-oss's native format) is a known quant", () => {
    expect(quantRow("MXFP4").bpp).toBe(0.55);
    expect(fit(card(16), moe20).runMode).toBe("gpu");
  });
});

describe("KV cache geometry", () => {
  const base: FitModel = { params: 27, layers: 62, kvHeads: 16, headDim: 128 };
  it("sliding-window layers stop growing at their window", () => {
    const full = kvCacheGB(base, 32768).gb;
    const gemma = kvCacheGB({ ...base, fullAttentionLayers: 10, slidingWindow: 1024 }, 32768).gb;
    expect(gemma).toBeLessThan(full / 4);
    // At a context shorter than the window, every layer holds the whole context.
    expect(kvCacheGB({ ...base, fullAttentionLayers: 10, slidingWindow: 1024 }, 512).gb).toBeCloseTo(kvCacheGB(base, 512).gb, 9);
  });

  it("linear or recurrent layers keep no growing cache", () => {
    const hybrid = kvCacheGB({ ...base, fullAttentionLayers: 16 }, 32768).gb;
    expect(hybrid).toBeCloseTo((kvCacheGB(base, 32768).gb * 16) / 62, 9);
  });

  it("latent attention caches its compressed width", () => {
    const mla = kvCacheGB({ params: 671, layers: 61, mlaKvWidth: 576 }, 8192);
    expect(mla.exact).toBe(true);
    expect(mla.gb).toBeCloseTo((576 * 61 * 8192 * 2) / GiB, 9);
  });

  it("a precomputed per-token figure wins and scales with the element size", () => {
    const m: FitModel = { ...base, kvBytesPerToken: { full: 100_000, sliding: 50_000 }, slidingWindow: 1024 };
    expect(kvCacheGB(m, 8192).gb).toBeCloseTo((100_000 * 8192 + 50_000 * 1024) / GiB, 9);
    expect(kvCacheGB(m, 8192, 1).gb).toBeCloseTo(kvCacheGB(m, 8192).gb / 2, 9);
  });

  it("q8_0 KV halves the cache", () => {
    expect(fit(card(24), base, { ctx: 16384, kvBytesPerElement: 1 }).kvBytes).toBeCloseTo(fit(card(24), base, { ctx: 16384 }).kvBytes / 2, 0);
  });
});

describe("reserve", () => {
  it("memory held back for the desktop can push a model off the card", () => {
    const m: FitModel = { params: 1, sizeBytes: 22.6 * GiB };
    expect(fit(card(24), m).runMode).toBe("gpu");
    expect(fit(card(24), m, { reserveBytes: GiB }).runMode).toBe("partial");
  });
  it("applies to unified memory too", () => {
    const mac: FitProfile = { ramBytes: 32 * GiB, cpuCores: 10, gpus: [{ model: "Apple M2 Pro", vramBytes: null }], bestVramBytes: null, unifiedMemory: true };
    expect(fit(mac, { params: 1 }, { unifiedUsable: 0.66, reserveBytes: 2 * GiB }).budgetBytes).toBeCloseTo((32 * 0.66 - 2) * GiB, 0);
  });
});

describe("layer-split offload speed (llmfit 1.1.18)", () => {
  const d32: FitModel = { params: 32.8, layers: 64, kvHeads: 8, headDim: 128 };
  it("a known card holding most layers beats one holding few", () => {
    const most = fit(card(16, 64), d32).tokPerSec!;
    const few = fit(card(6, 64), d32).tokPerSec!;
    expect(most).toBeGreaterThan(few);
  });
  it("never beats the all-on-GPU speed and never drops below system-memory speed", () => {
    const r = fit(card(12, 64), d32);
    const allGpu = fit(card(48, 64), d32).tokPerSec!;
    const ramOnly = 50 / (32.8 * 0.5);
    expect(r.runMode).toBe("partial");
    expect(r.tokPerSec!).toBeLessThan(allGpu);
    expect(r.tokPerSec!).toBeGreaterThan(ramOnly);
  });
  it("faster system memory makes it faster", () => {
    expect(fit(card(12, 64), d32, { ddrGBps: 100 }).tokPerSec!).toBeGreaterThan(fit(card(12, 64), d32).tokPerSec!);
  });
});

describe("quality estimate", () => {
  it("grows with size, continuously (no tier ties)", () => {
    const a = qualityEstimate({ params: 12 });
    const b = qualityEstimate({ params: 14 });
    expect(b).toBeGreaterThan(a);
    expect(qualityEstimate({ params: 70 })).toBeGreaterThan(qualityEstimate({ params: 32 }));
  });
  it("counts a mixture of experts as the geometric mean of total and active", () => {
    expect(effectiveParams(30, 3)).toBeCloseTo(Math.sqrt(90), 9);
    expect(effectiveParams(14)).toBe(14);
    expect(qualityEstimate({ params: 30, activeParams: 3 })).toBeCloseTo(qualityEstimate({ params: Math.sqrt(90) }), 9);
  });
  it("newer generations and recent releases score higher, and quantization costs points", () => {
    expect(qualityEstimate({ params: 14, family: "qwen", generation: 3.5 })).toBeGreaterThan(qualityEstimate({ params: 14, family: "qwen", generation: 2.5 }));
    expect(qualityEstimate({ params: 14, releaseMonth: "2026-09", nowMonth: "2026-10" })).toBeGreaterThan(qualityEstimate({ params: 14, releaseMonth: "2025-01", nowMonth: "2026-10" }));
    expect(qualityEstimate({ params: 14, quant: "Q8_0" })).toBeGreaterThan(qualityEstimate({ params: 14, quant: "Q4_K_M" }));
  });
  it("is not clamped at 100", () => {
    expect(qualityEstimate({ params: 400, family: "deepseek", generation: 4, quant: "Q8_0", releaseMonth: "2026-10", nowMonth: "2026-10" })).toBeGreaterThan(100);
  });
  it("ignores a malformed release month instead of guessing", () => {
    const plain = qualityEstimate({ params: 14 });
    for (const bad of ["2026-13", "2026-1", "26-01", "2026-01-05", "soon"]) {
      expect(qualityEstimate({ params: 14, releaseMonth: bad, nowMonth: "2026-10" }), bad).toBe(plain);
    }
  });
  it("quant names match in any case", () => {
    expect(quantRow("q8_0")).toEqual(quantRow("Q8_0"));
  });
});
