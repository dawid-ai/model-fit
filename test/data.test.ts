import { describe, expect, it } from "vitest";
import constants from "../data/constants.json" with { type: "json" };
import quant from "../data/quant.json" with { type: "json" };
import bandwidth from "../data/gpu-bandwidth.json" with { type: "json" };
import models from "../data/models.json" with { type: "json" };

// M5: every number in the data is finite, positive, and plausible — so a bad data PR fails here,
// naming the key, instead of showing a florist "Infinity words a second".
const ok = (v: unknown, lo: number, hi: number): boolean => typeof v === "number" && Number.isFinite(v) && v >= lo && v <= hi;

describe("M5: data sanity", () => {
  it("GPU bandwidths are 10–10,000 GB/s with a non-empty lowercase match", () => {
    const bad = bandwidth.gpus.filter((g) => !ok(g.gbps, 10, 10_000) || g.match.length === 0 || g.match !== g.match.toLowerCase());
    expect(bad.map((g) => g.match)).toEqual([]);
  });

  it("a more specific GPU name is never shadowed by an earlier, shorter one", () => {
    const shadowed: string[] = [];
    bandwidth.gpus.forEach((g, i) => {
      const earlier = bandwidth.gpus.slice(0, i).find((e) => g.match.includes(e.match));
      if (earlier) shadowed.push(`${g.match} shadowed by ${earlier.match}`);
    });
    expect(shadowed).toEqual([]);
  });

  it("quant rows are plausible", () => {
    const bad = Object.entries(quant.quants).filter(([, r]) => !ok(r.bpp, 0.1, 4) || !ok(r.speedBpp, 0.1, 4) || !ok(r.speedMultiplier, 0.1, 3));
    expect(bad.map(([k]) => k)).toEqual([]);
  });

  it("model specs are plausible and every Ollama tag is unique", () => {
    const bad = Object.entries(models.models).filter(
      ([, m]) => !ok(m.params, 0.01, 2000) || !ok(m.layers, 1, 200) || !ok(m.kvHeads, 1, 1024) || !ok(m.headDim, 1, 1024),
    );
    expect(bad.map(([k]) => k)).toEqual([]);
    const tags = Object.values(models.models).flatMap((m) => m.ollama);
    expect(new Set(tags).size).toBe(tags.length);
  });

  it("constants are plausible", () => {
    expect(ok(constants.efficiency, 0.1, 1)).toBe(true);
    expect(constants.levels.perfectMax < constants.levels.goodMax && constants.levels.goodMax < constants.levels.marginalMax).toBe(true);
    for (const [k, v] of Object.entries(constants.fallbackK)) expect(ok(v, 1, 1000), k).toBe(true);
  });
});
