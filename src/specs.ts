import modelTable from "../data/models.json" with { type: "json" };
import type { FitModel, ModelKind } from "./fit.js";

// Known model architectures, keyed by Hugging Face repo id AND by each Ollama tag. Lookups are an
// own-property check on a null-prototype map, so an untrusted key ("__proto__", "constructor", a
// 10 kB string) can only miss — never match a different model or reach the prototype.

export interface ModelSpec extends Required<Pick<FitModel, "params" | "layers" | "kvHeads" | "headDim">> {
  kind: ModelKind;
  /** The Hugging Face repo id this spec belongs to. */
  repo: string;
}

const MAX_KEY = 200;
const byKey: Record<string, ModelSpec> = Object.create(null) as Record<string, ModelSpec>;
for (const [repo, m] of Object.entries(modelTable.models)) {
  const spec: ModelSpec = { repo, params: m.params, layers: m.layers, kvHeads: m.kvHeads, headDim: m.headDim, kind: m.kind as ModelKind };
  byKey[repo.toLowerCase()] = spec;
  for (const tag of m.ollama) byKey[tag.toLowerCase()] = spec;
}

/** The spec for a Hugging Face repo id or an Ollama tag (case-insensitive), or null. */
export function specFor(key: unknown): ModelSpec | null {
  if (typeof key !== "string" || key.length === 0 || key.length > MAX_KEY) return null;
  const k = key.toLowerCase();
  return Object.hasOwn(byKey, k) ? byKey[k]! : null;
}
