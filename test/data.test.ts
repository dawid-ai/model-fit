import { describe, expect, it } from "vitest";
import constants from "../data/constants.json" with { type: "json" };
import quant from "../data/quant.json" with { type: "json" };
import bandwidth from "../data/gpu-bandwidth.json" with { type: "json" };
import models from "../data/models.json" with { type: "json" };

// M5: every number in the data is a finite, plausible number — so a bad data PR (a typo, a number
// written as a string) fails here, naming the key, instead of showing "Infinity words a second" or
// turning every model red.
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

  it("every quant row, including the default, is plausible", () => {
    const rows: [string, { bpp: unknown; speedBpp: unknown; speedMultiplier: unknown }][] = [["default", quant.default], ...Object.entries(quant.quants)];
    const bad = rows.filter(([, r]) => !ok(r.bpp, 0.1, 4) || !ok(r.speedBpp, 0.1, 4) || !ok(r.speedMultiplier, 0.1, 3));
    expect(bad.map(([k]) => k)).toEqual([]);
    expect(Object.hasOwn(quant.quants, constants.defaultQuant), "defaultQuant must be a known quant").toBe(true);
  });

  it("model specs are plausible and every Ollama tag is unique", () => {
    const bad = Object.entries(models.models).filter(
      ([, m]) => !ok(m.params, 0.01, 2000) || !ok(m.layers, 1, 200) || !ok(m.kvHeads, 1, 1024) || !ok(m.headDim, 1, 1024),
    );
    expect(bad.map(([k]) => k)).toEqual([]);
    const tags = Object.values(models.models).flatMap((m) => m.ollama);
    expect(new Set(tags).size).toBe(tags.length);
  });

  it("every constant is a plausible number", () => {
    const bounds: [string, unknown, number, number][] = [
      ["overheadGB", constants.overheadGB, 0, 8],
      ["kvBytesPerElement", constants.kvBytesPerElement, 0.25, 4],
      ["kvFallbackGBPerParamBPerToken", constants.kvFallbackGBPerParamBPerToken, 1e-7, 1e-3],
      ["defaultContext", constants.defaultContext, 256, 1_048_576],
      ["efficiency", constants.efficiency, 0.1, 1],
      ["manyCoresThreshold", constants.manyCoresThreshold, 1, 512],
      ["manyCoresBonus", constants.manyCoresBonus, 1, 2],
      ...Object.entries(constants.levels).map(([k, v]): [string, unknown, number, number] => [`levels.${k}`, v, 0.01, 1]),
      ...Object.entries(constants.runModeFactor).map(([k, v]): [string, unknown, number, number] => [`runModeFactor.${k}`, v, 0.01, 1]),
      ...Object.entries(constants.fallbackK).map(([k, v]): [string, unknown, number, number] => [`fallbackK.${k}`, v, 1, 1000]),
    ];
    expect(bounds.filter(([, v, lo, hi]) => !ok(v, lo, hi)).map(([k]) => k)).toEqual([]);
    const { perfectMax, goodMax, marginalMax } = constants.levels;
    expect(perfectMax < goodMax && goodMax < marginalMax).toBe(true);
  });
});
