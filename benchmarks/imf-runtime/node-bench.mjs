// E1: node-tier benchmark via the production npm registry path.
// usage: node node-bench.mjs <modelId> [...modelIds]
import { createHash } from "node:crypto"
import { performance } from "node:perf_hooks"

const { imf } = await import("interscript/ml")

for (const modelId of process.argv.slice(2)) {
  const hw = `${process.platform}/${process.arch} node ${process.version}`
  console.log(`\n=== ${modelId} (${hw}) ===`)

  // M1 cold: resolve index (Release + sidecar verify) + fetch + verify zip
  process.env["SECRYST_CACHE"] ??= undefined
  let t0 = performance.now()
  const resolved = await imf.resolve(modelId)
  const coldMs = performance.now() - t0
  console.log(`M1 cold resolve+fetch+verify: ${coldMs.toFixed(0)} ms (${(resolved.bytes.length / 1e6).toFixed(0)} MB)`)

  t0 = performance.now()
  await imf.resolve(modelId)
  console.log(`M1 warm (fs cache + re-verify): ${(performance.now() - t0).toFixed(0)} ms`)

  // M2 integrity tax: hash-only over the bytes (verify cost, isolated)
  t0 = performance.now()
  createHash("sha256").update(resolved.bytes).digest("hex")
  console.log(`M2 whole-file sha256: ${(performance.now() - t0).toFixed(0)} ms`)

  // M3 session create + M4 decode latency by length
  t0 = performance.now()
  const sample = {
    "ara-diac-small-1.0-int8": ["كتاب", "السلام عليكم ورحمة الله وبركاته", "السلام عليكم ورحمة الله وبركاته".repeat(8)],
    default: ["สวัสดี", "สวัสดีครับกรุงเทพมหานคร", "สวัสดีครับกรุงเทพมหานคร".repeat(8)],
  }[modelId] ?? { default: ["สวัสดี", "สวัสดีครับกรุงเทพมหานคร", "สวัสดีครับกรุงเทพมหานคร".repeat(8)] }["default"]
  const model = await imf.IMFModel.fromZipBytes(resolved.bytes)
  console.log(`M3 session create (zip open + member verify + ORT init): ${(performance.now() - t0).toFixed(0)} ms`)
  for (const [i, input] of sample.entries()) {
    const t = performance.now()
    const out = await model.translate(input)
    console.log(`M4 len=${input.length}B decode: ${(performance.now() - t).toFixed(0)} ms (out ${out.length}B)`)
  }
  const m = process.memoryUsage()
  console.log(`M5 rss: ${(m.rss / 1e6).toFixed(0)} MB`)
  await model.dispose?.()
}
