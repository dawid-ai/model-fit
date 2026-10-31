import constants from "../data/constants.json" with { type: "json" };
import type { FitModel } from "./fit.js";

const GiB = 1024 ** 3;

/** The KV cache a model holds for `ctx` tokens, in GiB, and whether it came from the model's own
 *  architecture (`exact`) or the params × context fallback.
 *
 *  Exact path (llmfit `kv_cache_gb`): per token and layer, 2 × KV heads × head size elements, or the
 *  latent width for multi-head latent attention (DeepSeek, Kimi). Only `fullAttentionLayers` hold
 *  the whole context. The other layers hold `slidingWindow` tokens when the model has one (Gemma,
 *  gpt-oss) and nothing when they're linear or recurrent (Qwen3-Next, Granite 4 hybrids).
 *  Sliding windows go beyond llmfit, which charges every layer the full context. */
export function kvCacheGB(model: FitModel, ctx: number, bytesPerElement: number = constants.kvBytesPerElement): { gb: number; exact: boolean } {
  const params = Math.max(model.params, 0.1);
  if (model.kvBytesPerToken !== undefined) {
    // Precomputed at fp16 (2 bytes per element); scale to the requested element size.
    const scale = bytesPerElement / 2;
    const windowTokens = model.slidingWindow !== undefined ? Math.min(ctx, model.slidingWindow) : ctx;
    const bytes = model.kvBytesPerToken.full * ctx + (model.kvBytesPerToken.sliding ?? 0) * windowTokens;
    return { gb: (bytes * scale) / GiB, exact: true };
  }
  const layers = model.layers;
  const width = model.mlaKvWidth ?? (model.kvHeads !== undefined && model.headDim !== undefined ? 2 * model.kvHeads * model.headDim : undefined);
  if (layers === undefined || width === undefined) {
    return { gb: constants.kvFallbackGBPerParamBPerToken * params * ctx * (bytesPerElement / constants.kvBytesPerElement), exact: false };
  }
  const full = Math.min(Math.max(model.fullAttentionLayers ?? layers, 0), layers);
  const windowed = layers - full;
  const windowTokens = model.slidingWindow !== undefined ? Math.min(ctx, model.slidingWindow) : 0;
  const tokens = full * ctx + windowed * windowTokens;
  return { gb: (width * tokens * bytesPerElement) / GiB, exact: true };
}
