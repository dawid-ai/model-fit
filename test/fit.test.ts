import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { fit, gpuBandwidthGBps, specFor, type FitModel, type FitProfile } from "../src/index.js";

const GiB = 1024 ** 3;
const here = path.dirname(fileURLToPath(import.meta.url));

interface Vector {
  source: string;
  input: { profile: FitProfile; model: FitModel & { id: string }; opts: { ctx: number } };
  expect: { needGB: number; tokPerSec: number | null; runMode: string };
}

// M1 — golden vectors. Most come from real `llmfit --json plan` runs; the rest are hand-derived from
// llmfit's formulas and say so in `source`. Any future port (Python) runs these same files.
const dir = path.join(here, "vectors");
const files = readdirSync(dir).filter((f) => f.endsWith(".json"));

describe("M1: golden vectors match llmfit", () => {
  it("has the coverage the contract asks for", () => {
    const all = files.flatMap((f) => JSON.parse(readFileSync(path.join(dir, f), "utf8")) as Vector[]);
    expect(all.filter((v) => v.source.startsWith("llmfit 1.1.16 plan --json")).length).toBeGreaterThanOrEqual(3);
    expect(new Set(all.map((v) => v.input.model.id)).size).toBe(9); // all 9 catalog models
  });

  for (const f of files) {
    const vectors = JSON.parse(readFileSync(path.join(dir, f), "utf8")) as Vector[];
    for (const v of vectors) {
      it(`${f} · ${v.input.model.id}`, () => {
        const r = fit(v.input.profile, v.input.model, v.input.opts);
        expect(r.runMode).toBe(v.expect.runMode);
        expect(Math.abs(r.needBytes / GiB - v.expect.needGB) / v.expect.needGB).toBeLessThanOrEqual(0.1);
        if (v.expect.tokPerSec === null) expect(r.tokPerSec).toBeNull();
        else expect(Math.abs(r.tokPerSec! - v.expect.tokPerSec) / v.expect.tokPerSec).toBeLessThanOrEqual(0.25);
      });
    }
  }
});

const gpu24: FitProfile = {
  ramBytes: 64 * GiB,
  cpuCores: 16,
  gpus: [{ model: "AMD Radeon RX 7900 XTX", vramBytes: 24 * GiB }],
  bestVramBytes: 24 * GiB,
  unifiedMemory: false,
};

describe("M2: fallbacks are honest", () => {
  it("a model with params only still gets a number, marked as estimated", () => {
    const r = fit(gpu24, { params: 7 });
    expect(r.needBytes).toBeGreaterThan(0);
    expect(r.estimated).toEqual({ weights: true, kv: true, speed: false });
  });

  it("an unknown GPU name still gets a speed, marked as estimated", () => {
    const r = fit({ ...gpu24, gpus: [{ model: "Mystery GPU", vramBytes: 24 * GiB }] }, { params: 3 });
    expect(r.tokPerSec).toBeGreaterThan(0);
    expect(r.estimated.speed).toBe(true);
  });

  it("a laptop GPU never borrows its desktop twin's bandwidth", () => {
    expect(gpuBandwidthGBps("NVIDIA GeForce RTX 4090 Laptop GPU")).toBeNull();
    expect(gpuBandwidthGBps("NVIDIA GeForce RTX 4090")).toBe(1008);
    expect(gpuBandwidthGBps("NVIDIA GeForce RTX 5070 Ti")).toBe(896); // the more specific entry wins
  });

  it("unknown memory and no GPU → won't run, no speed, never a guessed pass", () => {
    const r = fit({ ramBytes: null, cpuCores: null, gpus: [], bestVramBytes: null, unifiedMemory: false }, { params: 1 });
    expect(r.runMode).toBe("none");
    expect(r.tokPerSec).toBeNull();
    expect(r.share).toBe(Number.POSITIVE_INFINITY);
    expect(r.level).toBe("too-tight");
  });

  it("an exact file size replaces the params estimate", () => {
    const a = fit(gpu24, { params: 3, sizeBytes: 2_000_000_000 });
    expect(a.estimated.weights).toBe(false);
    expect(a.needBytes).toBeLessThan(fit(gpu24, { params: 3, sizeBytes: 3_000_000_000 }).needBytes);
  });

  it("non-text models get a memory verdict but no words-per-second", () => {
    const r = fit(gpu24, { params: 1, kind: "stt" });
    expect(r.runMode).toBe("gpu");
    expect(r.tokPerSec).toBeNull();
  });

  it("an iGPU below the useful floor counts as no GPU", () => {
    const igpu: FitProfile = { ramBytes: 16 * GiB, cpuCores: 8, gpus: [{ model: "Intel UHD", vramBytes: GiB }], bestVramBytes: GiB, unifiedMemory: false };
    expect(fit(igpu, { params: 0.5 }, { minUsefulVramBytes: 2 * GiB }).runMode).toBe("cpu");
    expect(fit(igpu, { params: 0.5 }).runMode).toBe("gpu"); // default floor 0: llmfit's behavior
  });

  it("unified memory honors a stricter usable share", () => {
    const mac: FitProfile = { ramBytes: 16 * GiB, cpuCores: 8, gpus: [{ model: "Apple M2", vramBytes: null }], bestVramBytes: null, unifiedMemory: true };
    const spec = specFor("qwen2.5:14b")!;
    expect(fit(mac, spec).runMode).toBe("gpu"); // all 16 GB usable (llmfit)
    const strict = fit(mac, spec, { unifiedUsable: 0.66 });
    expect(strict.level).toBe("marginal"); // ~9.8 GB of ~10.6 GB
  });
});

describe("M3: level boundaries", () => {
  // A 10 GiB pool and models whose need lands exactly on each boundary.
  const pool: FitProfile = { ramBytes: 64 * GiB, cpuCores: 8, gpus: [{ model: "x", vramBytes: 10 * GiB }], bestVramBytes: 10 * GiB, unifiedMemory: false };
  const sized = (needGB: number): FitModel => ({ params: 1, sizeBytes: (needGB - 0.5 - 0.000008 * 4096) * 1e9 });
  // Points sit a hair inside each side of a boundary — exactly ON 0.60/0.85/0.98 is float noise.
  it.each([
    [5.99, "perfect"],
    [6.01, "good"],
    [8.49, "good"],
    [8.51, "marginal"],
    [9.79, "marginal"],
  ] as const)("need %s GB of 10 GB → %s", (needGB, level) => {
    expect(fit(pool, sized(needGB)).level).toBe(level);
  });
  it("just over the marginal ceiling no longer fits the GPU", () => {
    expect(fit(pool, sized(9.81)).runMode).toBe("partial");
  });
  it("offload and CPU paths never read as perfect", () => {
    const cpu: FitProfile = { ramBytes: 64 * GiB, cpuCores: 8, gpus: [], bestVramBytes: null, unifiedMemory: false };
    expect(fit(cpu, { params: 1 }).level).toBe("good");
  });
});

describe("specFor: a lookup, never a sink", () => {
  it("resolves Hugging Face ids and Ollama tags, case-insensitively", () => {
    expect(specFor("Qwen/Qwen2.5-3B-Instruct")?.layers).toBe(36);
    expect(specFor("qwen2.5:3b")?.repo).toBe("Qwen/Qwen2.5-3B-Instruct");
    expect(specFor("LLAMA3.2:1B")?.headDim).toBe(64);
  });
  it.each(["qwen2.5:3b-evil", "__proto__", "constructor", "toString", "../x", "", "x".repeat(10_000), 42, null, {}])(
    "misses on %j",
    (k) => {
      expect(specFor(k)).toBeNull();
    },
  );
});
