import { readFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import express from "express";

const app = express();
const worker = process.env.INFERENCE_URL || "http://127.0.0.1:8000";
const port = Number(process.env.PORT || 3001);
const host = process.env.HOST || "127.0.0.1";
const omniTimeoutMs = Number(process.env.VLLM_OMNI_TIMEOUT_MS ?? 60_000);
if (!Number.isSafeInteger(omniTimeoutMs) || omniTimeoutMs < 1 || omniTimeoutMs > 600_000) {
  throw new Error("VLLM_OMNI_TIMEOUT_MS must be an integer between 1 and 600000.");
}

app.use((request, response, next) => {
  request.id = randomUUID();
  response.set("X-Request-ID", request.id);
  next();
});
app.use(express.json({ limit: "16kb" }));

function fail(request, response, status, message) {
  return response.status(status).json({ error: message, request_id: request.id });
}

app.get("/health", (request, response) => response.json({ status: "ready" }));

app.post(["/v1/speech", "/v1/speech/stream"], async (request, response) => {
  const { text, backend = "kokoro" } = request.body ?? {};
  if (!["kokoro", "vllm"].includes(backend)) return fail(request, response, 400, "Unknown backend.");
  const omni = backend === "vllm";
  const voice = request.body?.voice_id ?? (omni ? "ryan" : "af_heart");
  if (omni && !process.env.VLLM_OMNI_URL) return fail(request, response, 503, "Configure VLLM_OMNI_URL to enable vLLM-Omni.");
  if (typeof text !== "string" || !text.trim() || [...text].length > 2000) {
    return fail(request, response, 400, "Text must contain 1–2000 characters.");
  }
  if (typeof voice !== "string" || !(omni ? /^[a-z][a-z0-9_-]{0,63}$/i : /^a[fm]_[a-z]+$/).test(voice)) {
    return fail(request, response, 400, omni ? "Invalid Qwen speaker name." : "Use an American English voice ID such as af_heart.");
  }

  const stream = /\/stream\/?$/i.test(request.path);
  const timeout = AbortSignal.timeout(omni ? omniTimeoutMs : 60_000);
  const disconnected = new AbortController();
  response.on("close", () => {
    if (!response.writableFinished) disconnected.abort();
  });
  try {
    const upstream = await fetch(omni ? new URL("/v1/audio/speech", process.env.VLLM_OMNI_URL) : new URL(stream ? "/synthesize/stream" : "/synthesize", worker), {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Request-ID": request.id },
      body: JSON.stringify(omni ? { model: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", input: text, voice, language: "English", task_type: "CustomVoice", response_format: stream ? "pcm" : "wav", stream } : { text, voice_id: voice }),
      signal: AbortSignal.any([timeout, disconnected.signal]),
    });
    for (const header of ["Retry-After", "X-Queue-Wait-Ms", "X-Inference-Ms"]) {
      const value = upstream.headers.get(header);
      if (value !== null) response.set(header, value);
    }
    if (!upstream.ok) {
      const details = await upstream.json().catch(() => ({}));
      return fail(request, response, upstream.status, details.error || "Inference worker failed.");
    }
    if (stream) {
      response.set("Content-Type", "application/x-ndjson");
      response.set("Cache-Control", "no-store");
      await pipeline(Readable.fromWeb(upstream.body), ...(omni ? [pcmEvents] : []), response);
      return;
    }
    response.type("audio/wav").send(Buffer.from(await upstream.arrayBuffer()));
  } catch {
    if (disconnected.signal.aborted || response.destroyed) return;
    if (response.headersSent) {
      response.destroy();
      return;
    }
    const status = timeout.aborted ? 504 : 502;
    return fail(request, response, status, status === 504 ? "Inference timed out." : "Inference worker is unavailable.");
  }
});

async function* pcmEvents(source) {
  let tail = Buffer.alloc(0);
  let started = false;
  for await (const part of source) {
    const bytes = tail.length ? Buffer.concat([tail, part]) : Buffer.from(part);
    const even = bytes.length - bytes.length % 2;
    tail = bytes.subarray(even);
    if (!even) continue;
    if (!started) yield JSON.stringify({ type: "start", sample_rate: 24000 }) + "\n";
    started = true;
    yield JSON.stringify({ type: "audio", pcm: bytes.subarray(0, even).toString("base64") }) + "\n";
  }
  if (!started || tail.length) throw new Error("Incomplete upstream PCM");
  yield JSON.stringify({ type: "done" }) + "\n";
}

const experiments = new URL("../../experiments/", import.meta.url);

app.get("/v1/benchmarks/:phase", async (request, response) => {
  const { phase } = request.params;
  if (phase === "baseline") {
    const reports = await Promise.all([1, 2].map(async (concurrency) => {
      const path = new URL(`00_baseline/cpu-smoke-c${concurrency}.json`, experiments);
      return JSON.parse(await readFile(path, "utf8"));
    }));
    return response.json(reports);
  }
  if (!/^[a-z]+$/.test(phase)) return fail(request, response, 404, "Benchmark phase not found.");
  try {
    const path = new URL(`dashboard/${phase}.json`, experiments);
    return response.json(JSON.parse(await readFile(path, "utf8")));
  } catch (error) {
    if (error.code === "ENOENT") return fail(request, response, 404, "No measurements published.");
    throw error;
  }
});

app.use((request, response) => fail(request, response, 404, "Route not found."));
app.use((error, request, response, next) => {
  if (response.headersSent) return next(error);
  if (error.type === "entity.too.large") return fail(request, response, 413, "Request is too large.");
  if (error.type === "entity.parse.failed") return fail(request, response, 400, "Invalid JSON request.");
  return fail(request, response, 500, "Unexpected gateway error.");
});

app.listen(port, host, () => console.log(`VoiceForge listening on http://${host}:${port}`));
