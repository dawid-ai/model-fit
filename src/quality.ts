import { quantRow } from "./fit.js";

// A rough quality estimate for a model with no benchmark score, ported from llmfit
// `quality_score` (fit.rs:2160) with three fixes:
//   - the size curve is continuous (log-interpolated between llmfit's tier anchors), so two
//     models of different size never tie just because they share a tier;
//   - a mixture-of-experts model counts as the geometric mean of its total and active size (a
//     30B-A3B model behaves like a ~9.5B dense one, not llmfit's 3.3B);
//   - no clamp at 100, so new large models don't all tie at the top.
// Apps with a real benchmark score should prefer it; this is the fallback.

export interface QualityInput {
  /** Billions of parameters (all of them). */
  params: number;
  /** Billions active per token, for a mixture-of-experts model. */
  activeParams?: number;
  /** Model family, for example "qwen3", "gemma3", "llama3.3" (lowercase name or architecture). */
  family?: string;
  /** Model generation within its family, for example 3.5 for Qwen3.5 (0 or absent when unknown). */
  generation?: number;
  /** Release month, "YYYY-MM". */
  releaseMonth?: string;
  /** "Now", "YYYY-MM", passed in so the estimate stays pure. */
  nowMonth?: string;
  quant?: string;
}

// llmfit's tier anchors: (billions, points). Below the first and above the last, the curve is flat.
const ANCHORS: [number, number][] = [
  [0.5, 30],
  [2, 45],
  [5, 60],
  [8.5, 75],
  [15, 82],
  [30, 89],
  [70, 95],
  [400, 100],
];

/** Effective dense size in billions: a mixture-of-experts model counts as √(total × active). */
export function effectiveParams(params: number, activeParams?: number): number {
  const total = Math.max(params, 0.01);
  if (activeParams === undefined || activeParams >= total) return total;
  return Math.sqrt(total * Math.max(activeParams, 0.01));
}

function sizePoints(b: number): number {
  if (b <= ANCHORS[0]![0]) return ANCHORS[0]![1];
  for (let i = 1; i < ANCHORS.length; i++) {
    const [x1, y1] = ANCHORS[i]!;
    if (b <= x1) {
      const [x0, y0] = ANCHORS[i - 1]!;
      return y0 + ((Math.log(b) - Math.log(x0)) / (Math.log(x1) - Math.log(x0))) * (y1 - y0);
    }
  }
  return ANCHORS[ANCHORS.length - 1]![1];
}

// llmfit fit.rs:2192, first substring match.
const FAMILY_BONUS: [string, number][] = [
  ["qwen", 2],
  ["deepseek", 3],
  ["llama", 2],
  ["mistral", 1],
  ["mixtral", 1],
  ["gemma", 1],
  ["phi", 0],
  ["starcoder", 1],
];

function monthsBetween(from: string, to: string): number | null {
  const a = /^(\d{4})-(\d{2})$/.exec(from);
  const b = /^(\d{4})-(\d{2})$/.exec(to);
  if (!a || !b) return null;
  return (Number(b[1]) - Number(a[1])) * 12 + (Number(b[2]) - Number(a[2]));
}

/** Estimated quality points (llmfit's scale: ~30 for a 0.5B model, ~95 for a 70B one, plus bonuses). */
export function qualityEstimate(m: QualityInput): number {
  const base = sizePoints(effectiveParams(m.params, m.activeParams));
  const family = m.family?.toLowerCase() ?? "";
  const familyBonus = FAMILY_BONUS.find(([k]) => family.includes(k))?.[1] ?? 0;
  const gen = Math.min(Math.max(((m.generation ?? 0) - 1) * 3, 0), 9);
  const age = m.releaseMonth && m.nowMonth ? monthsBetween(m.releaseMonth, m.nowMonth) : null;
  const recency = age === null || age < 0 ? 0 : age < 3 ? 3 : age < 9 ? 1.5 : 0;
  return base + familyBonus + gen + recency + quantRow(m.quant).qualityPenalty;
}
