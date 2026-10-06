import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// M4 (security): the published code does no I/O and has no dependencies — so any app can run it in
// any process (even a sandboxed one) without granting it anything. Scans the BUILT output.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FORBIDDEN = [
  /from\s+["']node:/,
  /from\s+["'](fs|child_process|net|http|https|dgram|dns|tls|worker_threads|os)["']/,
  /import\(\s*["']/,
  /require\(/,
  /\bfetch\s*\(/,
  /\bXMLHttpRequest\b/,
  /\bWebSocket\b/,
  /\bprocess\./,
];

function jsFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? jsFiles(path.join(dir, e.name)) : e.name.endsWith(".js") ? [path.join(dir, e.name)] : [],
  );
}

describe("M4: purity", () => {
  it("has zero runtime dependencies", () => {
    const pkg = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8")) as Record<string, unknown>;
    expect(pkg["dependencies"] ?? {}).toEqual({});
    expect(pkg["peerDependencies"] ?? {}).toEqual({});
    expect(pkg["optionalDependencies"] ?? {}).toEqual({});
  });

  it("the built code imports nothing but its own modules and data", () => {
    const dist = path.join(root, "dist", "src");
    expect(existsSync(dist), "run the build first").toBe(true);
    const files = jsFiles(dist);
    expect(files.length).toBeGreaterThan(0);
    const offenders: string[] = [];
    for (const f of files) {
      const src = readFileSync(f, "utf8");
      for (const re of FORBIDDEN) if (re.test(src)) offenders.push(`${path.relative(root, f)}: ${re}`);
      for (const m of src.matchAll(/from\s+["']([^"']+)["']/g)) {
        if (!m[1]!.startsWith("./") && !m[1]!.startsWith("../")) offenders.push(`${path.relative(root, f)}: imports ${m[1]}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
