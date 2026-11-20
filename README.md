# VoiceForge — Voice AI Inference Optimization Engine

> **Status:** Local Kokoro inference, an Express gateway, and a browser demo.
>
> **Goal:** Build and benchmark a production-minded streaming text-to-speech (TTS) serving system that reduces time-to-first-audio, improves concurrent throughput, and controls GPU cost without unacceptable speech-quality regressions.

## Repository setup

Clone the repository and copy the environment template for local development.

```sh
git clone https://github.com/dne-labs/forgeVoice.git
cd forgeVoice
cp .env.example .env
```

The environment template documents local ports for the planned gateway and worker.

## Run with Docker

Docker Compose builds and runs the worker, gateway, and web app together. This
requires Docker Engine with Compose v2 and enough disk space for the PyTorch
image and the Kokoro weights.

From the repository root:

```sh
docker compose up --build
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000). Compose publishes each
service on loopback only:

| Service | Container port | Host address |
|---|---:|---|
| Web | 3000 | `http://127.0.0.1:3000` |
| Gateway | 3001 | `http://127.0.0.1:3001` |
| Worker | 8000 | `http://127.0.0.1:8000` |

The first build creates a Python 3.12 worker image with eSpeak NG, PyTorch
2.8.0 CPU, and Kokoro 0.9.4. Kokoro-82M weights download when the worker starts,
and the default voice pack downloads on first synthesis, both into the
`voiceforge_hf-cache` volume. Later starts reuse the cache, so leave extra time
for the first `docker compose up`. `docker compose down` keeps the cache; add
`-v` to delete it.

`docker-compose.gpu.yml` switches the worker to the CUDA 12.6 PyTorch wheel and
requests all GPUs. It requires an NVIDIA driver and the NVIDIA Container
Toolkit:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The web image bakes the gateway address into the Next.js build. The default is
`http://gateway:3001`; set `WEB_GATEWAY_URL` and rebuild the `web` service to use
another address. The gateway receives `INFERENCE_URL=http://worker:8000`.

To opt into Qwen speech through vLLM-Omni, set `VLLM_OMNI_URL` when starting
Compose. The gateway container can reach a server on the host through
`host.docker.internal`:

```sh
VLLM_OMNI_URL=http://host.docker.internal:8001 docker compose up
```

Useful commands:

```sh
docker compose ps                 # container and health status
docker compose logs -f worker     # follow worker logs
docker compose restart gateway    # restart one service
docker compose down               # stop the stack, keep the model cache
docker compose down -v            # stop and remove the model cache
```

The native-process instructions below remain available for development without
Docker.

## Generate a WAV with Kokoro

These smoke-test commands target x86-64 Linux and Python 3.12. Install
[uv](https://docs.astral.sh/uv/) and eSpeak NG (`sudo pacman -S --needed espeak-ng`
on Arch or `sudo apt-get install -y espeak-ng` on Ubuntu).

From the repository root, run this on your local machine:

```sh
cd services/inference
uv run --locked --python 3.12 \
  --index https://download.pytorch.org/whl/cpu \
  --with 'torch==2.8.0+cpu' --with 'kokoro==0.9.4' \
  voiceforge-synthesize --device cpu
```

For an NVIDIA GPU in Colab, enable a GPU runtime, clone the repo, install uv
and eSpeak NG, then run this from `services/inference` in a shell or
`%%bash` notebook cell:

```sh
uv run --locked --python 3.12 \
  --index https://download.pytorch.org/whl/cu126 \
  --with 'torch==2.8.0+cu126' --with 'kokoro==0.9.4' \
  voiceforge-synthesize --device cuda --output /content/output.wav
```

uv installs the pinned model runtime and matching PyTorch wheel for each command.
The first run downloads the Python dependencies, Kokoro-82M weights, and voice
preset. Both commands use the text “Hello, welcome to VoiceForge.” and the
American English voice `af_heart`. Use `--text` and `--voice` to change them.
The result is a mono, 24 kHz, signed 16-bit PCM WAV. CPU runs write
`output.wav` by default. See the [project plan](VOICEFORGE_PROJECT_PLAN.md) for
the worker and benchmarking milestones.

## Run the local web app

Start the Python worker in one terminal from the repository root:

```sh
cd services/inference
uv run --locked --python 3.12 \
  --index https://download.pytorch.org/whl/cpu \
  --with 'torch==2.8.0+cpu' --with 'kokoro==0.9.4' \
  voiceforge-worker --device cpu
```

Start the gateway in a second terminal from the repository root:

```sh
cd apps/api
npm ci
npm start
```

Start the Next.js app in a third terminal from the repository root:

```sh
cd apps/web
npm ci
npm run dev
```

Use Node.js 22 or newer and open [http://127.0.0.1:3000](http://127.0.0.1:3000).
Open **Text to Speech** to choose a voice and generate, play, or download
a WAV. **Benchmarks** reads the recorded inference reports from the gateway.
The overview reflects saved browser activity. Projects, favorites and preferences
persist locally, and generated clips are saved to browser IndexedDB. Voice
previews and full generations use Kokoro. Voice cloning, transcription, dubbing,
billing, API key management, and advanced voice controls are marked Coming soon.
The gateway accepts `POST /v1/speech` and forwards requests to the worker's
`POST /synthesize`. The worker loads the model once and accepts one active
synthesis at a time. A bounded FIFO queue accepts four additional requests by
default. A full queue returns HTTP 503 with `Retry-After: 1`. Requests that
wait longer than 15 seconds receive HTTP 504.
The worker runs on port 8000, the gateway on 3001, and Next.js on 3000.
The native processes bind to loopback. In Docker, the worker and gateway bind to
`0.0.0.0` inside the container network while Compose publishes them on loopback.
To use a different gateway address, set
`VOICEFORGE_API_URL` before starting Next.js. If building for production, set
it before `npm run build` as well. The gateway uses `INFERENCE_URL` to locate
the worker.

### Frontend structure

The Next.js App Router uses TypeScript and React components in `apps/web`.
Tailwind design tokens and the light/dark palettes live in `app/globals.css`.
Shared motion variants are in `lib/motion.ts`, and workspace state is in
`lib/workspace.tsx`.

| Location | Responsibility |
|---|---|
| `app/page.tsx`, `app/[section]/page.tsx` | Dashboard and workspace routes |
| `components/page-shell.tsx`, `sidebar.tsx`, `topbar.tsx` | Layout, navigation, theme and notifications |
| `components/workspace-overlays.tsx` | Command palette, dialogs and toasts |
| `components/dashboard-home.tsx` | Local overview and recent generations |
| `components/speech-studio.tsx`, `generate-bar.tsx`, `voice-card.tsx` | Script editing, generation, and voice picker |
| `components/audio-player-row.tsx`, `lib/audio-store.ts` | Playback, waveform, and local clip persistence |
| `components/voice-library.tsx`, `records-table.tsx`, `settings-page.tsx` | Local voices, projects, history, and settings |
| `components/benchmarks.tsx`, `benchmark-charts.tsx` | Measured baseline and optimization reports |

### Stream audio as it is generated

Choose **Stream as generated** in the speech studio to start playback as
Kokoro finishes each segment. **Stop** cancels the browser request and playback.
The worker checks for disconnects between segments, so an in-progress model
call can still finish before the worker becomes available again.

The streaming endpoint is `POST /v1/speech/stream`. It returns newline-delimited
JSON with a `start` event (sample rate), base64 PCM `audio` events, and a `done`
or `error` event. The browser schedules mono 16-bit PCM with Web Audio and
builds a downloadable WAV after a successful stream.

To repeat the paired streaming benchmark through the running Next.js app:

```sh
python experiments/01_streaming/run.py --pairs 5 \
  --setup "Describe your hardware, runtime, and worker settings" \
  --output /tmp/voiceforge-streaming.json
```

The report measures time to a complete playable payload and full response time.
The first metric ends at HTTP delivery and excludes browser decoding, scheduling,
and physical playback. Use the same hardware, workload, and voice for both modes.
The runner uses a newline-delimited prompt so both delivery modes synthesize
the same text segments. Raw trials include generated audio duration and
streaming chunk counts.

### Scheduling

Set worker flags `--queue-limit`, `--queue-timeout`, and `--queue-policy` to
change admission. `--queue-limit 0` reproduces immediate rejection of overlapping
requests. The experimental `shortest` policy uses text length as its estimate
and gives the oldest request priority after two seconds of waiting.
The queue deadline applies before inference starts. Disconnects remove queued
requests and stop streaming between model chunks. A running model call retains
the inference slot until it yields or completes.

The service sends `X-Queue-Wait-Ms` on successful responses. Full WAV includes
`X-Inference-Ms`, and streaming includes `inference_ms` in the final event.
The Express gateway applies a 60-second upstream deadline.

To repeat the CPU scheduling benchmark, run from the repository root:

```sh
cd services/inference
uv run --locked --python 3.12 \
  --index https://download.pytorch.org/whl/cpu \
  --with 'torch==2.8.0+cpu' --with 'kokoro==0.9.4' \
  python ../../experiments/02_scheduling/run.py --output /tmp/voiceforge-queueing.json
```

The benchmark sends identical bursts of short and long requests to the HTTP
worker under immediate rejection, FIFO, and shortest-first admission.
**Benchmarks → Queueing** displays outcomes, tail latency, and queue wait.

### Probe batching feasibility

The batching experiment runs two prompts serially, then passes a single row,
two equal-length rows, and two padded rows to Kokoro's token-forward method.
The result is shown under **Benchmarks → More phases → Batching**.

On Kokoro 0.9.4 with PyTorch 2.8.0 CPU, both prompts produced audio serially.
The two-row calls, including the equal-length control, failed at duration
alignment because the repeat counts must be scalar or one-dimensional.
Batching is deferred for this backend. See the
[recorded feasibility report](experiments/dashboard/batching.json).

```sh
cd services/inference
uv run --locked --python 3.12 \
  --index https://download.pytorch.org/whl/cpu \
  --with 'torch==2.8.0+cpu' --with 'kokoro==0.9.4' \
  python ../../experiments/03_batching/run.py --output /tmp/voiceforge-batching.json
```

### Cache voice conditioning

The worker keeps reusable Kokoro voice packs on the model device in a
least-recently-used cache with a 4 MiB tensor-data budget and a 10-minute TTL.
Run the worker with `--conditioning-cache-bytes` or
`--conditioning-cache-ttl` (seconds) to change those limits. A zero-byte
budget reloads packs without retaining them. TTL runs from each load and
cache entries are scoped to one loaded adapter. Restart the worker after
changing model, language, device, or precision.

View the worker counters locally with
`curl -fsS http://127.0.0.1:8000/conditioning-cache`. Resident bytes cover
retained pack data. In-flight holds and allocator overhead can add memory.

Kokoro already caches voice packs in host memory. The benchmark compares that
existing warm path with managed cold and warm modes using the same model and
seed. Prime the model and the default `af_heart` voice first using the CPU
synthesis command above. Then run:

```sh
cd services/inference
HF_HUB_OFFLINE=1 uv run --locked --python 3.12 \
  --index https://download.pytorch.org/whl/cpu \
  --with 'torch==2.8.0+cpu' --with 'kokoro==0.9.4' \
  python ../../experiments/04_conditioning_cache/run.py --output /tmp/voiceforge-caching.json
```

**Benchmarks → More phases → Conditioning cache** shows the CPU measurements.
In eight CPU rounds, median first PCM was 667 ms with Kokoro's warm cache
and 668 ms with the managed warm cache. The small run showed no clear CPU
latency gain. All seeded outputs matched byte for byte. CUDA transfer savings
require a separate GPU run.

### Run GPU precision in Colab

Open the [Phase 6 notebook](https://colab.research.google.com/github/dne-labs/Voice-inference/blob/voiceforge-gpu-precision/experiments/05_precision/colab.ipynb),
select a GPU runtime, and run all cells. The notebook primes model files first,
then runs the same prompts and voice on the same GPU. It compares FP32 with
FP16 autocast and tests BF16 only when PyTorch reports native support. A failed
candidate is recorded with its error.

The downloaded ZIP contains `precision.json`, matching WAV samples, and a
whole-run `nvidia-smi` utilization log sampled once a second. Compare the WAV
samples for pronunciation and timbre. The runner records duration changes,
clipping, allocator peak memory, model load time, and first-PCM and synthesis
latency. It accepts an optional `--usd-per-hour` to estimate compute cost per
audio minute using a known hourly rate. Initial downloads and idle time are
excluded from that estimate.

A completed CUDA report can be placed at `experiments/dashboard/precision.json`
for **Benchmarks → More phases → GPU precision**. CPU smoke results are kept
outside the dashboard.

The CUDA worker also accepts `--cuda-min-free-mib`. Before synthesis, it
checks available device memory and returns HTTP 503 with `Retry-After: 1`
below that threshold. The default is zero. Calibrate a positive threshold
with measurements from the target GPU and workload. Free memory can change
after the check.

### Use the benchmark dashboard

Open **Benchmarks** to explore P50/P95 full-response latency, success rate,
and audio throughput for the recorded CPU runs. Latency percentiles only
include successful requests. The dashboard shows sample counts and explains
why the two concurrency runs cannot establish a speedup.

Each subsequent optimization has a phase view. To publish a measured
before/after comparison, add `experiments/dashboard/<phase>.json`.
Use `streaming`, `queueing`, `batching`, `caching`, `precision`, `chunking`,
or `observability` as the filename stem. Include the hardware, workload,
model configuration, concurrency, and sample sizes in `setup`. Measure
both variants under matching conditions and link raw results in the report
directory for reproducibility. The schema is:

```text
setup: string
notes: optional string
metrics: list of { name: string, unit: string, before: number, after: number,
                   better: "lower" | "higher" }
```

Use non-negative measured values. Phases without a report display their
planned metrics and an awaiting-data status.

### Measure the local baseline

Start the worker with `--queue-limit 0` to reproduce the original immediate-
rejection baseline. With the worker and gateway running, run from the repo root:

```sh
python experiments/00_baseline/run.py --url http://127.0.0.1:3001/v1/speech --concurrency 1 --output /tmp/voiceforge-c1.json
python experiments/00_baseline/run.py --url http://127.0.0.1:3001/v1/speech --concurrency 2 --output /tmp/voiceforge-c2.json
```

Reports include each request's total latency, generated audio duration, and
status. The runner warms up the service before measuring. With no waiting slots,
overlapping requests can return HTTP 503. These runs measure full WAV responses.

## CI and releases

GitHub Actions runs the repository checks on pull requests and pushes to `main`.
Pushing a `v*` tag runs the same checks and publishes a source archive and
SHA-256 checksum to GitHub Releases. Releases package the tracked source files.
No AWS resources are provisioned.

## 1. Why this project exists

A basic TTS demo calls a model and returns a WAV file. **VoiceForge** investigates the harder systems problems around serving TTS to many simultaneous users: streaming, scheduling, batching, reusable conditioning, GPU memory limits, request cancellation, tail latency, and cost.

The portfolio outcome is a reproducible engineering case study for voice-AI inference roles (for example, companies building real-time speech products). It is **not** a guarantee of hiring or a claim to reproduce any company's private infrastructure.

### Success criteria

1. A working local HTTP API and streaming endpoint backed by an open-source TTS model.
2. A reproducible CPU baseline, followed by GPU baselines on AWS if eligible.
3. A load generator with controlled workloads and saved raw benchmark results.
4. At least **two measured optimizations** against the same baseline and hardware.
5. Clear trade-off reports: p50/p95/p99 latency, throughput, audio quality, GPU memory, and cost.
6. Public architecture diagrams, experiment methodology, failure cases, and setup instructions.

**Important:** Do not promise a 2× speedup. Measure first, optimize, then publish what actually happens.

## 2. Constraints and deployment strategy

- **Development computer:** Arch Linux, currently without an NVIDIA GPU.
- **Cloud:** AWS account with *approximately* $100 in credits; coverage, expiry, GPU quotas, and region pricing must be verified before launch.
- **Budget-first approach:** Build API, scheduler, tests, dashboards, and synthetic benchmarks locally. Rent GPU only for short, scheduled experiments.
- **Model:** Start with **one** open-source TTS model, tentatively Kokoro, after checking its license, dependencies, supported streaming/chunking behavior, and CPU/GPU compatibility. Do not implement generic batching before proving the chosen model supports batched inference.
- **Cloud:** A `g4dn.xlarge` (NVIDIA T4, 16 GiB VRAM) is a possible starting point **subject to availability, quota, pricing, and credit eligibility**; not a guaranteed option.
- **Avoid 24/7 hosting.** Use short-lived instances and persist benchmark artifacts separately.

## 3. System architecture

```text
Client / load generator
         |
         v
TypeScript API gateway (Express)
  - validation, request IDs, limits
  - HTTP generate + streaming endpoint
         |
         v
Inference scheduler
  - queue, admission control, deadlines
  - optional microbatching (when model supports it)
         |
         v
Python TTS inference worker (PyTorch)
  - model adapter
  - conditioning / synthesis / audio encoding
  - chunk events + cancellation checks
         |
         v
Audio stream -> gateway -> client

Observability: structured logs + metrics + benchmark JSON
Optional later: Redis for shared state/cache and Prometheus/Grafana
```

**Design choice:** Begin with a single-process scheduler and one worker. Add Redis/distributed coordination only when experiments justify it. The API-to-worker protocol should be explicit and versioned.

## 4. Technology stack

| Layer | Initial choice | Why |
|---|---|---|
| Gateway | TypeScript, Express, Bun or Node | Familiar APIs and streaming control |
| Inference | Python, PyTorch | TTS model ecosystem |
| Model | Kokoro (candidate) | Small-model baseline; verify behavior/license |
| Transport | HTTP for full audio; WebSocket or HTTP chunked stream for streaming | Compare end-to-end delivery |
| Scheduler | TypeScript initially | Easy policy experiments |
| Cache | In-process first; Redis later | Avoid premature infrastructure |
| Benchmarks | Python or TypeScript load generator + JSON/CSV | Reproducible comparisons |
| Metrics | Prometheus/Grafana later | Latency and GPU dashboards |
| GPU | AWS EC2 when permitted | Short, controlled experiments |
| Packaging | Docker Compose locally | Repeatable development |

## 5. Proposed repository structure

This is the **target layout**. Create directories incrementally as phases require them; don't scaffold empty complexity on day one.

```text
voiceforge/
├── README.md                         # Problem, quick start, results
├── PROJECT_PLAN.md                   # This roadmap
├── LICENSE
├── .env.example                      # Safe configuration placeholders
├── .gitignore
├── package.json                      # Workspace scripts
├── bun.lock                          # If Bun chosen
├── apps/
│   ├── api/
│   │   ├── src/
│   │   │   ├── index.ts              # Server entry
│   │   │   ├── routes/
│   │   │   │   ├── synthesize.ts      # POST /v1/speech
│   │   │   │   └── stream.ts          # WS or streaming HTTP endpoint
│   │   │   ├── middleware/
│   │   │   │   ├── validation.ts
│   │   │   │   └── request-id.ts
│   │   │   ├── clients/
│   │   │   │   └── inference-client.ts
│   │   │   └── telemetry/
│   │   │       └── metrics.ts
│   │   └── tests/
│   └── dashboard/                     # Optional: build in later phase
│       └── ...
├── services/
│   ├── inference/
│   │   ├── pyproject.toml
│   │   ├── src/voiceforge_inference/
│   │   │   ├── server.py             # Worker protocol endpoints
│   │   │   ├── models/
│   │   │   │   ├── base.py           # TTS model interface
│   │   │   │   └── kokoro.py         # First adapter
│   │   │   ├── synthesis.py          # Model execution
│   │   │   ├── audio.py              # Format and chunk handling
│   │   │   ├── conditioning.py       # If supported by model
│   │   │   └── metrics.py
│   │   └── tests/
│   └── scheduler/
│       ├── src/
│       │   ├── queue.ts
│       │   ├── fifo.ts
│       │   ├── admission.ts
│       │   ├── batcher.ts            # Only after batching feasibility test
│       │   ├── deadlines.ts
│       │   └── cancellation.ts
│       └── tests/
├── packages/
│   ├── protocol/                      # Shared request/event schemas
│   └── benchmark/
│       ├── workloads/                 # Short/medium/long, multi-voice
│       ├── runner.py
│       ├── metrics.py
│       └── report.py
├── infra/
│   ├── docker/
│   │   └── compose.yaml
│   ├── aws/
│   │   ├── README.md                 # Launch/stop checklist
│   │   └── bootstrap.sh              # Optional GPU environment setup
│   └── monitoring/                    # Later: Prometheus/Grafana
├── experiments/
│   ├── 00_baseline/
│   ├── 01_streaming/
│   ├── 02_scheduling/
│   ├── 03_batching/
│   ├── 04_conditioning_cache/
│   ├── 05_precision/
│   └── 06_adaptive_chunking/
├── reports/
│   ├── methodology.md
│   ├── results.md
│   └── figures/
└── scripts/
    ├── dev.sh
    ├── bench-local.sh
    └── bench-gpu.sh
```

## 6. API and worker contracts (initial proposal)

### Full synthesis

`POST /v1/speech`

```json
{
  "text": "Hello, this is VoiceForge.",
  "voice_id": "default",
  "format": "wav"
}
```

Response: `audio/wav` with `X-Request-ID`. For long tasks, a job-based API may be more appropriate; choose after testing.

### Streaming synthesis

`WS /v1/speech/stream` (proposed; final transport chosen in Phase 2).

Messages should include: `request_id`, `started`, `audio_chunk` (binary audio with sequence metadata), `completed`, `error`, and cancellation. A chunk must carry format/sample rate and sequence index so the client can decode correctly. **A chunked WAV file is not automatically playable streaming audio**; define the audio framing/codec and test real playback.

### Internal worker protocol

- `SynthesizeRequest`: ID, text, voice, model, output format, deadline, priority.
- `AudioChunk`: ID, sequence number, bytes, sample rate, codec, timestamp.
- `SynthesisFinished`: ID, model timings, total generated audio duration, status.
- `CancelRequest`: ID and reason.

Define validation, size limits, errors, and timeouts. Keep external client API separate from internal model-specific parameters.

## 7. Step-by-step implementation plan

### Phase 0 — Verify feasibility and control costs (Day 1)

**Tasks**

- [ ] Confirm AWS credit program, remaining balance, expiration, and eligible EC2 services.
- [ ] Check regional GPU quota and current instance pricing; do not assume quota is available.
- [ ] Create AWS Budget alerts and document the stop/terminate procedure.
- [ ] Pick one TTS model and record license, CPU/GPU requirements, audio format, streaming support, and batch support.
- [ ] Define benchmark workload: 10 short prompts, 10 medium, 10 long; consistent voice, output sample rate, seeds where supported.

**Files:** `README.md`, `PROJECT_PLAN.md`, `reports/methodology.md`, `infra/aws/README.md`, `packages/benchmark/workloads/`.

**Done when:** AWS risks are documented and the model runs at least one local CPU inference (or a documented reason why local execution is infeasible).

### Phase 1 — Minimal inference and truthful baseline (Days 2–5)

**Tasks**

- [ ] Create Python worker with `base.py` model interface and first adapter.
- [ ] Expose a local worker endpoint and `POST /v1/speech` through Express.
- [ ] Implement request IDs, structured errors, input length limits, and basic concurrency guard.
- [ ] Write tests for empty/oversized text, model failure, timeouts, and audio format.
- [ ] Build load generator for concurrency = 1, 2, 4, 8 (as supported).
- [ ] Save raw latency measurements, audio duration, output format, CPU/RAM usage.
- [ ] Run short AWS GPU baseline **only after** credits and quota checks.

**Files:** `services/inference/...`, `apps/api/...`, `packages/benchmark/runner.py`, `experiments/00_baseline/`.

**Done when:** One command synthesizes audio and one command generates a reproducible baseline report.

### Phase 2 — Real streaming and cancellation (Days 6–9)

**Tasks**

- [ ] Inspect whether model produces incremental audio natively. If not, clearly label segmented synthesis and its quality/latency limitations.
- [ ] Define binary chunk framing and client playback behavior.
- [ ] Implement stream start, audio chunks, completion, error, and cancellation.
- [ ] Add bounded buffers and backpressure handling for slow clients.
- [ ] Measure time-to-first-*playable*-audio and chunk gaps under load.
- [ ] Compare full-file delivery versus actual streaming.

**Files:** `apps/api/src/routes/stream.ts`, `services/inference/src/voiceforge_inference/audio.py`, `experiments/01_streaming/`.

**Done when:** A client starts playing before the entire utterance is generated, if the model supports incremental output; otherwise publish the limitation and benchmark segmented output separately.

### Phase 3 — Scheduling, fairness, and tail latency (Days 10–13)

**Tasks**

- [ ] Implement bounded FIFO queue and maximum concurrent worker slots.
- [ ] Add queue timeout, deadline, overload rejection, cancellation, and no-starvation tests.
- [ ] Compare FIFO against a documented latency-aware policy.
- [ ] Track queue wait, execution time, total response time, p50/p95/p99 TTFA.
- [ ] Run a mixed short/long request workload.

**Files:** `services/scheduler/src/{queue,fifo,admission,deadlines,cancellation}.ts`, `experiments/02_scheduling/`.

**Done when:** Overload is controlled and benchmark results show the scheduling trade-offs.

### Phase 4 — Batching (Days 14–17; conditional)

**Prerequisite:** Verify the chosen TTS model supports **real batched inference**. Grouping requests in a queue without a batched model forward pass is not GPU batching.

**Tasks**

- [ ] Add a batch-capability test to the Python adapter.
- [ ] If supported: implement batch size, max-wait, compatible voice/model grouping, and padding strategy.
- [ ] Compare no batching, fixed batching, and adaptive microbatching under identical load.
- [ ] Measure throughput, TTFA p95/p99, GPU memory, and utilization.
- [ ] If unsupported: document the result and either optimize scheduling or add a second batch-capable adapter later.

**Files:** `services/scheduler/src/batcher.ts`, `services/inference/src/voiceforge_inference/models/base.py`, `experiments/03_batching/`.

**Done when:** Results distinguish throughput gains from added queueing delay; no invented speedups.

### Phase 5 — Reusable voice conditioning cache (Days 18–20; model-dependent)

**Tasks**

- [ ] Identify expensive, reusable conditioning artifacts supported by the chosen model.
- [ ] Key cache by model version, voice ID, relevant preprocessing/config, and embedding version.
- [ ] Add bounded memory, TTL/invalidation, and cache hit/miss metrics.
- [ ] Test same-voice repeated requests against cold-start requests.
- [ ] Confirm audio quality and speaker identity are unchanged within tolerance.

**Files:** `services/inference/src/voiceforge_inference/conditioning.py`, `experiments/04_conditioning_cache/`.

**Done when:** Cache reduces measured repeated-work overhead, **or** report why the model offers no useful conditioning reuse. Redis is optional.

### Phase 6 — GPU precision and memory-aware serving (Days 21–24)

**Tasks**

- [ ] Capture baseline GPU peak VRAM, utilization, model load time, and inference timing.
- [ ] Test supported BF16/FP16/FP32 configurations; test INT8 only if the model/toolchain genuinely supports it.
- [ ] Track any numerical, pronunciation, timbre, or stability regressions.
- [ ] Add memory-aware admission limits and test overload behavior.
- [ ] Compare cost per generated audio minute at fixed workload and GPU price.

**Files:** `experiments/05_precision/`, `reports/results.md`.

**Done when:** A table compares performance, memory, quality, and cost for supported configurations. INT4 is an optional research extension, not a required feature.

### Phase 7 — Adaptive chunking (Days 25–27; conditional)

**Tasks**

- [ ] Determine whether the model supports safe chunk-size changes without breaking prosody.
- [ ] Compare a few supported chunk policies under varying load.
- [ ] Measure first playable audio, gap duration, boundary artifacts, and throughput.
- [ ] Reject policies that lower TTFA but cause audible degradation.

**Files:** `experiments/06_adaptive_chunking/`, `services/inference/src/voiceforge_inference/audio.py`.

**Done when:** Trade-offs are quantified; if model-native chunking is fixed, document why adaptive chunking is not applicable.

### Phase 8 — Observability, reproducibility, and portfolio (Days 28–30+)

**Tasks**

- [ ] Export request, queue, inference, and GPU metrics.
- [ ] Add a lightweight dashboard (Grafana or Next.js, not necessarily both).
- [ ] Automate benchmark runs with saved configuration, model commit, instance type, precision, and warmup.
- [ ] Publish raw data and before/after plots with confidence/variance where possible.
- [ ] Add architecture diagram, demo video, failure analysis, and cost breakdown.
- [ ] Write a short case study: hypothesis → implementation → experiment → result → trade-off.

**Files:** `infra/monitoring/`, `apps/dashboard/`, `reports/`, `README.md`.

**Done when:** Another engineer can reproduce results or understand precisely why they cannot.

## 8. Benchmark methodology

### Required metrics

| Metric | Definition | Why it matters |
|---|---|---|
| TTFA | Client request sent → first *playable* audio | Perceived responsiveness |
| End-to-end latency | Request sent → last audio chunk received | Total completion time |
| Inter-chunk gap | Playback gap between consecutive usable chunks | Stream smoothness |
| RTF | Synthesis compute time / generated audio duration | < 1 is faster than real time |
| Throughput | Generated audio seconds / wall-clock second | Server capacity |
| p50/p95/p99 | Percentiles over individual requests | Tail behavior |
| Queue wait | Time between admission and worker start | Scheduler pressure |
| Peak VRAM | Maximum allocated/used GPU memory | Capacity limits |
| Cost/audio minute | Estimated compute+relevant infra cost / audio minutes | Unit economics |
| Audio quality | Listening tests + selected objective checks | Avoid bad optimizations |

**Important experimental rules:** same model/version, same prompts, same voice, same instance type, same warmup, same concurrency and request-arrival pattern, and repeated trials. Record cold-start separately from steady state. Compare end-to-end and model-only timings; do not mix them. For streaming, network conditions and audio decoding affect client TTFA.

### Workload matrix

- Prompt lengths: short (~1 sentence), medium (~1 paragraph), long (~several paragraphs).
- Concurrency: 1, 2, 4, 8, then increase only if resources permit.
- Arrival pattern: steady requests and burst traffic.
- Voice patterns: one repeated voice and multiple voices (if model supports them).
- Warm cache vs cold cache (when caching exists).

### Example results table (fill only after measurements)

| Experiment | TTFA p95 | Throughput | Peak VRAM | Quality notes | Cost/audio min |
|---|---:|---:|---:|---|---:|
| Baseline | TBD | TBD | TBD | TBD | TBD |
| Streaming | TBD | TBD | TBD | TBD | TBD |
| Scheduling | TBD | TBD | TBD | TBD | TBD |
| Microbatching | TBD | TBD | TBD | TBD | TBD |
| Cache | TBD | TBD | TBD | TBD | TBD |
| Precision | TBD | TBD | TBD | TBD | TBD |

## 9. AWS budget and safety checklist

**Before first GPU launch**

- [ ] Verify credits apply to the selected EC2 instance, region, storage, and network charges.
- [ ] Verify credits expire after the planned experiment period.
- [ ] Verify quota and availability for GPU family in target region.
- [ ] Set billing budget alerts; alerts are **not** hard spending caps.
- [ ] Set a strict personal experiment timebox (e.g. 2–4 hours).
- [ ] Keep security groups restrictive; don't expose inference or SSH broadly.
- [ ] Use least-privilege credentials and never commit AWS keys.

**After every experiment**

- [ ] Save raw results, logs, environment metadata, and plots.
- [ ] Terminate GPU instance when finished; stopping may still incur EBS charges.
- [ ] Check leftover EBS volumes, snapshots, elastic IPs, and other billable resources.
- [ ] Review AWS cost dashboard; billing updates can lag.

**Illustrative allocation for ~$100 credits (not guaranteed):** ~$55 GPU experiments, ~$15 storage/network, ~$15 repeat tests, ~$15 reserve. Actual rates and eligibility must be checked live.

## 10. Explicitly out of scope for v1

- Training a TTS foundation model from scratch.
- Building a general-purpose GPU kernel compiler.
- Claiming native streaming from a model that only supports full-utterance synthesis.
- Multi-region autoscaling, Kubernetes, or elaborate distributed queues before one-worker benchmarks.
- Production-grade voice cloning or user-uploaded voice datasets.
- Claiming optimizations such as speculative decoding, KV-cache quantization, or FlashAttention apply to every TTS architecture; applicability depends on the actual model.

## 11. First coding milestone: exact order

1. Initialize repository and write `README.md` + `.env.example`.
2. Create `services/inference` Python environment and a model-adapter interface.
3. Load the selected model on CPU and synthesize a single test sentence to WAV.
4. Add `POST /synthesize` to the Python worker.
5. Add `apps/api` Express gateway forwarding requests to the worker.
6. Add validation, request IDs, errors, and a concurrency limit.
7. Write one smoke test and one end-to-end test.
8. Implement benchmark runner for 1 and 2 concurrent requests.
9. Record CPU baseline in `experiments/00_baseline/`.
10. Only then evaluate AWS GPU readiness and perform the first GPU baseline.

**Rule for implementation:** Complete one milestone, run tests, record observations, then proceed. Do not create a large untested skeleton and call it finished.

## 12. Learning topics alongside the build

- TTS model architecture: acoustic generation, vocoder, conditioning, autoregressive vs non-autoregressive.
- GPU serving: warmup, model loading, VRAM, mixed precision, utilization.
- Queueing theory: arrival rate, service time, tail latency, backpressure, admission control.
- Batching: actual batched tensor computation, padding, microbatch wait, fairness.
- Streaming audio: PCM, WAV containers, codec framing, sample rates, playback buffering.
- Experimental design: reproducibility, baseline selection, p95/p99, variance, quality evaluation.

## 13. Final GitHub deliverables

- [ ] Source code and tests.
- [ ] Local development instructions and GPU deployment instructions.
- [ ] API/worker protocol documentation.
- [ ] Benchmark tool, workload definitions, and raw result files.
- [ ] At least two measured performance comparisons.
- [ ] Clear quality evaluation and failure analysis.
- [ ] Architecture diagram and dashboard screenshots.
- [ ] Demo showing concurrent requests and streaming playback.
- [ ] Technical report: **“Engineering a Low-Latency Voice Inference Serving System Under a $100 GPU Budget.”**

---

**Starting point:** Phase 0, then Phase 1. Treat the rest as a prioritized plan that may change based on model capabilities and measured bottlenecks.
