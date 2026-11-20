"use client";

import { useEffect, useState } from "react";
import BenchmarkChart, { chartColors, ReportCard } from "./benchmark-charts";
import { SectionHeading } from "./ui";

type Trial = { status: number | null; elapsed_seconds: number; audio_seconds?: number; prompt_index: number };
type Run = { concurrency: number; successes: number; failures: number; p50_seconds: number | null; p95_seconds: number | null; audio_seconds_per_wall_second: number; trials: Trial[] };
type Metric = { name: string; unit: string; before: number; after: number; better: "lower" | "higher" };
type PairRun = { first_ms: number; complete_ms: number; chunk_count?: number; audio_seconds: number };
type Policy = {
  id: string; label: string; completed: number; rejected: number; expired: number; failed: number;
  first_p50_ms: number | null; first_p95_ms: number | null; first_p99_ms: number | null;
  queue_p95_ms: number | null; inference_p50_ms: number | null; total_p95_ms: number | null;
  short_first_p50_ms: number | null;
};
type CacheMode = { label: string; first_pcm_p50_ms: number; total_p50_ms: number };
type CacheStats = { hits: number; misses: number; resident_bytes: number; entries: number };
type Report = { modes?: CacheMode[]; managed_warm?: CacheStats; policies?: Policy[]; setup: string; notes?: string; metrics: Metric[]; variants?: { wav: PairRun[]; stream: PairRun[] } };
type Feasibility = {
  kind: "batching-feasibility";
  status: "unsupported" | "needs-validation";
  setup: string;
  notes: string;
  serial: { prompt: string; token_count: number; waveform_samples: number; forward_calls: number }[];
  checks: { label: string; input_shape: number[]; forward_calls: number; outcome: string; error?: string }[];
};
type PrecisionRun = {
  label: string; status: "measured" | "skipped" | "failed"; reason?: string;
  p50_first_pcm_ms?: number; peak_allocated_mib?: number; peak_reserved_mib?: number;
  audio_seconds_per_second?: number; estimated_cost_per_audio_minute?: number | null;
};
type PrecisionReport = Report & {
  kind: "precision-benchmark"; device: "cuda"; gpu: string;
  model_load_seconds: number; runs: PrecisionRun[];
};
type Result = { id: string; data?: Run[] | Report | Feasibility | PrecisionReport; status?: "missing" | "error" };

const otherPhases = [
  ["batching", "Batching", "Check whether one model forward can synthesize multiple requests."],
  ["caching", "Conditioning cache", "Cache hit rate, warm latency, and model memory."],
  ["precision", "GPU precision", "Peak VRAM, latency, throughput, and listening quality."],
  ["chunking", "Adaptive chunking", "Time to first playable payload and gaps between chunks."],
  ["chunkingrepeat", "Chunking repeat", "First audio across three prompts in a fresh GPU session."],
  ["chunkingload", "Chunking under load", "Compare paragraph and sentence streaming at 1, 2, and 4 concurrent requests."],
  ["vllmstreaming", "vLLM streaming", "T4 first-audio latency, throughput, and completion under concurrent load."],
  ["observability", "Reproducibility", "Variation across repeated trials and cost per audio minute."],
];

const variants = [
  { key: "wav", label: "Full WAV", color: chartColors.neutral },
  { key: "stream", label: "Streaming", color: chartColors.accent },
];
const number = (value: number) => new Intl.NumberFormat("en", { maximumFractionDigits: 2 }).format(value);
const seconds = (ms: number) => `${number(ms / 1000)} s`;

function Metrics({ values }: { values: { label: string; value: string; detail: string }[] }) {
  return <dl className="overview-metrics benchmark-metrics">
    {values.map(({ label, value, detail }) => <div key={label}><dt>{label}</dt><dd>{value}</dd><small>{detail}</small></div>)}
  </dl>;
}

function DetailsTable({ title, headers, rows }: { title: string; headers: string[]; rows: (string | number)[][] }) {
  return <details className="panel benchmark-details">
    <summary>{title}</summary>
    <div className="benchmark-table-scroll"><table>
      <thead><tr>{headers.map((header) => <th key={header} scope="col">{header}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, i) => <td key={i}>{cell}</td>)}</tr>)}</tbody>
    </table></div>
  </details>;
}

function Baseline({ runs }: { runs: Run[] }) {
  const single = runs.find(({ concurrency }) => concurrency === 1);
  const outcomes = runs.map((run) => ({ label: `Concurrency ${run.concurrency}`, successful: run.successes, failed: run.failures }));
  const latencies = runs.map((run) => ({ label: `Concurrency ${run.concurrency}`, p50: run.p50_seconds, p95: run.p95_seconds }));
  const throughput = runs.map((run) => ({ label: `Concurrency ${run.concurrency}`, audio: run.audio_seconds_per_wall_second }));
  const trials = runs.flatMap((run) => run.trials.map((trial, index) => [
    run.concurrency, index + 1, trial.prompt_index + 1, trial.status ?? "Client error", `${number(trial.elapsed_seconds)} s`,
  ]));
  return <>
    {single && <Metrics values={[
      { label: "Median response", value: single.p50_seconds === null ? "—" : `${number(single.p50_seconds)} s`, detail: "Concurrency 1" },
      { label: "Successful requests", value: `${single.successes} / ${single.successes + single.failures}`, detail: "Concurrency 1" },
      { label: "Audio throughput", value: `${number(single.audio_seconds_per_wall_second)}×`, detail: "Audio seconds per wall-clock second" },
    ]} />}
    <div className="benchmark-grid">
      <BenchmarkChart title="Request outcomes" description="Successful and failed requests in each run." kind="stacked" data={outcomes}
        series={[{ key: "successful", label: "Succeeded", color: chartColors.accent }, { key: "failed", label: "Failed", color: chartColors.rejected }]} />
      <BenchmarkChart title="Response latency" description="Successful requests only. Concurrent latency is based on one result." data={latencies} unit=" s"
        series={[{ key: "p50", label: "P50", color: chartColors.accent }, { key: "p95", label: "P95", color: chartColors.neutral }]} />
      <BenchmarkChart title="Audio throughput" description="Seconds of generated audio per wall-clock second." data={throughput} unit="×"
        series={[{ key: "audio", label: "Audio throughput", color: chartColors.accent }]} />
    </div>
    <p className="benchmark-note">Shared CPU, Kokoro 0.9.4, voice af_heart. At concurrency 2, five of six requests returned HTTP 503. The percentiles describe this small smoke test.</p>
    <DetailsTable title="View raw request results" headers={["Concurrency", "Request", "Prompt", "Status", "Elapsed"]} rows={trials} />
  </>;
}

function Streaming({ report }: { report: Report }) {
  const first = report.metrics.find((metric) => metric.name.includes("playable"));
  const complete = report.metrics.find((metric) => metric.name.includes("complete"));
  const paired = report.variants;
  if (!first || !complete || !paired) return <Comparison report={report} />;

  const summary = [
    { label: "First payload", wav: first.before / 1000, stream: first.after / 1000 },
    { label: "Complete", wav: complete.before / 1000, stream: complete.after / 1000 },
  ];
  const pairs = paired.wav.slice(0, paired.stream.length).map((wav, index) => ({ label: `Run ${index + 1}`, wav, stream: paired.stream[index] }));
  const delta = first.before ? ` vs. ${seconds(first.before)} for full WAV` : "";
  return <>
    <Metrics values={[
      { label: "First playable payload", value: seconds(first.after), detail: `Streaming${delta}` },
      { label: "Complete response", value: seconds(complete.after), detail: `Streaming vs. ${seconds(complete.before)} for full WAV` },
      { label: "Paired runs", value: String(pairs.length), detail: "Same prompt, voice, and CPU" },
    ]} />
    <div className="benchmark-grid">
      <BenchmarkChart title="Median delivery time" description="Full WAV and streaming, compared at both milestones." data={summary} series={variants} unit=" s" />
      <BenchmarkChart title="First playable payload by run" description="Elapsed time until the client receives a playable WAV or the first PCM chunk." kind="line" unit=" s" series={variants}
        data={pairs.map(({ label, wav, stream }) => ({ label, wav: wav.first_ms / 1000, stream: stream.first_ms / 1000 }))} />
      <BenchmarkChart title="Complete response by run" description="Time from request to receipt of the complete response." kind="line" unit=" s" series={variants}
        data={pairs.map(({ label, wav, stream }) => ({ label, wav: wav.complete_ms / 1000, stream: stream.complete_ms / 1000 }))} />
    </div>
    <ReportNotes report={report} />
    <DetailsTable title="View paired measurements" headers={["Run", "WAV first", "Stream first", "WAV complete", "Stream complete", "Chunks"]}
      rows={pairs.map(({ label, wav, stream }) => [label, seconds(wav.first_ms), seconds(stream.first_ms), seconds(wav.complete_ms), seconds(stream.complete_ms), stream.chunk_count ?? "—"])} />
  </>;
}

function Scheduling({ report }: { report: Report }) {
  const policies = report.policies;
  if (!policies?.length) return <Comparison report={report} />;
  const admitted = policies.filter(({ id }) => id !== "reject");
  const toSeconds = (value: number | null) => value === null ? null : value / 1000;
  const format = (value: number | null) => value === null ? "—" : seconds(value);
  return <>
    <div className="benchmark-grid mt-4">
      <BenchmarkChart title="Request outcomes" description="The same burst workload under all three admission policies." kind="stacked"
        data={policies.map(({ label, completed, rejected, expired, failed }) => ({ label, completed, rejected, expired: expired + failed }))}
        series={[
          { key: "completed", label: "Completed", color: chartColors.accent },
          { key: "rejected", label: "Queue full", color: chartColors.rejected },
          { key: "expired", label: "Expired or failed", color: chartColors.neutral },
        ]} />
      <BenchmarkChart title="Time to first playable payload" description="Successful requests under the two queue policies." unit=" s"
        data={admitted.map((policy) => ({ label: policy.label, p50: toSeconds(policy.first_p50_ms), p95: toSeconds(policy.first_p95_ms), p99: toSeconds(policy.first_p99_ms) }))}
        series={[
          { key: "p50", label: "P50", color: chartColors.accent },
          { key: "p95", label: "P95", color: chartColors.neutral },
          { key: "p99", label: "P99", color: chartColors.rejected },
        ]} />
      <BenchmarkChart title="Short requests and queue wait" description="Short-request median first payload and overall P95 queue wait." unit=" s"
        data={admitted.map((policy) => ({ label: policy.label, short: toSeconds(policy.short_first_p50_ms), waiting: toSeconds(policy.queue_p95_ms) }))}
        series={[
          { key: "short", label: "Short request P50", color: chartColors.accent },
          { key: "waiting", label: "Queue wait P95", color: chartColors.neutral },
        ]} />
    </div>
    <ReportNotes report={report} />
    <DetailsTable title="View scheduling measurements" headers={["Policy", "Completed", "Queue wait P95", "Inference P50", "Total response P95"]}
      rows={policies.map((policy) => [policy.label, policy.completed, format(policy.queue_p95_ms), format(policy.inference_p50_ms), format(policy.total_p95_ms)])} />
  </>;
}

function Conditioning({ report }: { report: Report }) {
  if (!report.modes?.length) return <Comparison report={report} />;
  const stats = report.managed_warm;
  const samples = report.modes.map((mode) => ({ label: mode.label, first: mode.first_pcm_p50_ms, total: mode.total_p50_ms }));
  return <>
    {stats && <Metrics values={[
      { label: "Managed cache hits", value: number(stats.hits), detail: "Repeated voice requests" },
      { label: "Managed cache misses", value: number(stats.misses), detail: "Includes the initial pack load" },
      { label: "Resident conditioning", value: `${number(stats.resident_bytes / 1024)} KiB`, detail: `${stats.entries} cached voice pack(s)` },
    ]} />}
    <div className="benchmark-grid mt-4">
      <BenchmarkChart title="Time to first PCM" description="Median time until the first audio chunk reaches the adapter." data={samples} unit=" ms"
        series={[{ key: "first", label: "P50 first PCM", color: chartColors.accent }]} />
      <BenchmarkChart title="Total synthesis time" description="Median time until all audio chunks are ready." data={samples} unit=" ms"
        series={[{ key: "total", label: "P50 synthesis", color: chartColors.neutral }]} />
    </div>
    <ReportNotes report={report} />
  </>;
}

function ReportNotes({ report }: { report: Report }) {
  return <div className="benchmark-note"><p>{report.setup}</p>{report.notes && <p>{report.notes}</p>}</div>;
}

function Comparison({ report, labels = ["Before", "After"] }: { report: Report; labels?: [string, string] }) {
  return <>
    <div className="benchmark-grid">{report.metrics.map((metric) => <BenchmarkChart key={metric.name} title={metric.name} description={metric.better === "lower" ? "Lower is better." : "Higher is better."}
      data={[{ label: labels[0], value: metric.before }, { label: labels[1], value: metric.after }]} unit={metric.unit}
      series={[{ key: "value", label: "Measured value", color: chartColors.accent }]} />)}</div>
    <ReportNotes report={report} />
  </>;
}

function Precision({ report }: { report: PrecisionReport }) {
  const format = (value: number | null | undefined, unit: string) => value == null ? "—" : `${number(value)}${unit}`;
  return <>
    <ReportCard title={report.gpu} description={`Model load: ${format(report.model_load_seconds, " s")}`}>
      <p className="benchmark-note">WAV samples are required to assess pronunciation and timbre.</p>
    </ReportCard>
    <DetailsTable title="View precision run results" headers={["Mode", "Outcome", "P50 first PCM", "Peak allocated", "Peak reserved", "Audio / time", "Cost / audio min"]}
      rows={report.runs.map((run) => [
        run.label, run.reason ? `${run.status}: ${run.reason}` : run.status,
        format(run.p50_first_pcm_ms, " ms"), format(run.peak_allocated_mib, " MiB"),
        format(run.peak_reserved_mib, " MiB"), format(run.audio_seconds_per_second, "×"),
        run.estimated_cost_per_audio_minute == null ? "—" : `$${run.estimated_cost_per_audio_minute.toFixed(6)}`,
      ])} />
    {report.metrics.length ? <Comparison report={report} /> : <ReportNotes report={report} />}
  </>;
}

function hasPrecision(value: unknown): value is PrecisionReport {
  if (!value || typeof value !== "object") return false;
  const report = value as PrecisionReport;
  return report.kind === "precision-benchmark" && report.device === "cuda" && typeof report.gpu === "string" &&
    typeof report.setup === "string" && Number.isFinite(report.model_load_seconds) &&
    Array.isArray(report.runs) && report.runs.some((run) => run?.status === "measured") &&
    Array.isArray(report.metrics) && (!report.metrics.length || hasReport(report));
}

function Batching({ report }: { report: Feasibility }) {
  const blocked = report.status === "unsupported";
  return <>
    <ReportCard title="Batching feasibility" description={blocked ? "The tested Kokoro backend failed both two-row probes." : "A multi-row output still needs audio and output-length validation."}>
      <p className="benchmark-note">{report.setup}</p>
      <p className="benchmark-note">{report.notes}</p>
    </ReportCard>
    <DetailsTable title="View serial controls" headers={["Prompt", "Tokens", "Audio samples", "Forward calls"]}
      rows={report.serial.map((item) => [item.prompt, item.token_count, item.waveform_samples, item.forward_calls])} />
    <DetailsTable title="View batch probes" headers={["Probe", "Input shape", "Forward calls", "Outcome"]}
      rows={report.checks.map((check) => [check.label, check.input_shape.join(" × "), check.forward_calls, check.error ?? check.outcome])} />
  </>;
}

function hasFeasibility(value: unknown): value is Feasibility {
  if (!value || typeof value !== "object") return false;
  const report = value as Feasibility;
  return report.kind === "batching-feasibility" && ["unsupported", "needs-validation"].includes(report.status) &&
    typeof report.setup === "string" && typeof report.notes === "string" && Array.isArray(report.serial) &&
    Array.isArray(report.checks) && report.checks.every((check) => Array.isArray(check?.input_shape));
}

function hasReport(value: unknown): value is Report {
  if (!value || typeof value !== "object") return false;
  const report = value as Report;
  return typeof report.setup === "string" && Array.isArray(report.metrics) && report.metrics.length > 0 &&
    report.metrics.every((metric) => typeof metric?.name === "string" && typeof metric.unit === "string" &&
      [metric.before, metric.after].every((number) => Number.isFinite(number) && number >= 0) && ["lower", "higher"].includes(metric.better));
}

export default function Benchmarks() {
  const [id, setId] = useState("streaming");
  const [report, setReport] = useState<Result | null>(null);
  const [attempt, retry] = useState(0);
  const current = report?.id === id ? report : null;
  const phase = otherPhases.find(([key]) => key === id);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch(`/v1/benchmarks/${id}`, { signal: controller.signal });
        if (response.status === 404 && id !== "baseline" && id !== "streaming" && id !== "queueing") {
          if (!controller.signal.aborted) setReport({ id, status: "missing" });
          return;
        }
        if (!response.ok) throw new Error("Benchmark request failed");
        const data = await response.json();
        const valid = id === "baseline" ? Array.isArray(data)
          : id === "precision" ? hasPrecision(data)
          : hasReport(data) || (id === "batching" && hasFeasibility(data));
        if (!valid) throw new Error("Invalid benchmark data");
        if (!controller.signal.aborted) setReport({ id, data });
      } catch {
        if (!controller.signal.aborted) setReport({ id, status: "error" });
      }
    }
    load();
    return () => controller.abort();
  }, [id, attempt]);

  function select(next: string) {
    if (next === id) return;
    setReport(null);
    setId(next);
  }

  return <section aria-label="Inference benchmarks" className="benchmark-page">
    <SectionHeading title="Benchmarks" description="Recorded latency, throughput, and reliability measurements." />
    <div className="benchmark-toolbar">
      <nav aria-label="Benchmark reports" className="benchmark-tabs">
        <button type="button" aria-pressed={id === "streaming"} onClick={() => select("streaming")}>Streaming</button>
        <button type="button" aria-pressed={id === "baseline"} onClick={() => select("baseline")}>CPU baseline</button>
        <button type="button" aria-pressed={id === "queueing"} onClick={() => select("queueing")}>Queueing</button>
      </nav>
      <select className="field compact-field" aria-label="More benchmark phases" value={phase ? id : ""} onChange={(event) => select(event.target.value || "streaming")}>
        <option value="">More phases</option>
        {otherPhases.map(([key, label]) => <option value={key} key={key}>{label}</option>)}
      </select>
    </div>
    {!current && <div className="skeleton" role="status" aria-label="Loading benchmark report" />}
    {current?.status === "missing" && <ReportCard title={`${phase?.[1]} · Coming soon`} description={phase?.[2] ?? "No measurements have been published."}><p className="benchmark-note">No measured report is available for this phase yet.</p></ReportCard>}
    {current?.status === "error" && <ReportCard title="Report unavailable" description="Check that the local Express gateway is running."><button type="button" className="button mt-3" onClick={() => { setReport(null); retry((value) => value + 1); }}>Retry</button></ReportCard>}
    {current?.data && (Array.isArray(current.data) ? <Baseline runs={current.data} />
      : "kind" in current.data && current.data.kind === "batching-feasibility" ? <Batching report={current.data} />
      : "kind" in current.data && current.data.kind === "precision-benchmark" ? <Precision report={current.data} />
      : id === "streaming" ? <Streaming report={current.data} />
      : id === "queueing" ? <Scheduling report={current.data} />
      : id === "caching" ? <Conditioning report={current.data} />
      : <Comparison report={current.data} labels={id === "vllmstreaming" ? ["1 concurrent", "2 concurrent"] : id.startsWith("chunking") ? ["Paragraph", "Sentence"] : undefined} />)}
  </section>;
}
