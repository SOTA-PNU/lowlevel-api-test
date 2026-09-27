"""Build the static results dashboard from triton_test.py JSON reports.

Usage: python3 report/build_site.py --data DIR --out DIR [--fragment]

Every *.json report under --data is loaded. The latest report per hardware id
feeds the status, performance and functional sections; all reports feed the
performance trends. Standard library only.
"""
import argparse
import html
import json
import pathlib
from datetime import datetime, timezone

BACKEND_ORDER = {"cpu": 0, "cuda": 1, "npu": 2}
# Product names for chip models reported by rbln-smi.
HW_ALIASES = {
    "npu-rbln-ca22": "ATOM+ (RBLN-CA22)",
    "npu-rbln-ca25": "ATOM-Max (RBLN-CA25)",
}

def load_reports(data_dir):
    reports = []
    for path in sorted(pathlib.Path(data_dir).rglob("*.json")):
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip {path}: {exc}")
            continue
        if isinstance(report, dict) and report.get("schema_version") == 1:
            reports.append(report)
    return reports

def short_name(hw):
    if hw["id"] in HW_ALIASES:
        return HW_ALIASES[hw["id"]]
    if hw["backend"] == "cpu":
        return "CPU"
    return hw["model"].removeprefix("NVIDIA ")

def history_entry(report):
    return {
        "generated_at": report["generated_at"],
        "run": report["run"],
        "versions": report["versions"],
        "summary": report["summary"],
        "perf": {
            r["name"]: {"ms": r["ms"], "gbps": r["gbps"], "ops_per_s": r["ops_per_s"], "result": r["result"]}
            for r in report["results"] if r["module"] == "perf"
        },
    }

def build_payload(reports, site=None):
    by_hw = {}
    for report in reports:
        by_hw.setdefault(report["hardware"]["id"], []).append(report)
    hardware = []
    for hw_id, items in by_hw.items():
        # One entry per run: a re-run of the same run id replaces the earlier attempt.
        runs = {}
        for report in sorted(items, key=lambda r: r["generated_at"]):
            runs[report["run"].get("run_id") or report["generated_at"]] = report
        ordered = sorted(runs.values(), key=lambda r: r["generated_at"])
        latest = ordered[-1]
        hw = dict(latest["hardware"], name=short_name(latest["hardware"]))
        hardware.append({
            "hardware": hw,
            "latest": {k: latest[k] for k in ("generated_at", "run", "versions", "config", "summary", "results")},
            "history": [history_entry(r) for r in ordered],
        })
    hardware.sort(key=lambda h: (BACKEND_ORDER.get(h["hardware"]["backend"], 9), h["hardware"]["id"]))
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site": site,
        "hardware": hardware,
    }

TEMPLATE = r"""<title>Triton 연산자 테스트 보드</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">
<style>
:root {
  color-scheme: light;
  --bg: #f6f7f8;
  --surface: #ffffff;
  --surface-2: #eef0f2;
  --ink: #111418;
  --ink-2: #4a5360;
  --muted: #737c88;
  --line: #dde1e6;
  --line-strong: #c4cad2;
  --accent: #0f6e8c;
  --accent-soft: #e2f0f4;
  --good: #006300;
  --good-soft: #e3f3e3;
  --bad: #b3261e;
  --bad-soft: #fbe6e4;
  --warn: #8a5a00;
  --warn-soft: #fcf0d6;
  --none: #737c88;
  --none-soft: #eef0f2;
  --s1: #2a78d6;
  --s2: #eb6834;
  --s3: #1baf7a;
  --s4: #eda100;
  --grid: #e8ebee;
  --shadow: 0 1px 2px rgba(17, 20, 24, 0.06);
  --sans: "IBM Plex Sans KR", system-ui, -apple-system, "Segoe UI", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #0e1114;
    --surface: #161a1f;
    --surface-2: #1d232a;
    --ink: #f2f4f6;
    --ink-2: #b7c0ca;
    --muted: #8a939d;
    --line: #2a3139;
    --line-strong: #3a434d;
    --accent: #4fb3d4;
    --accent-soft: #15303a;
    --good: #3fc43f;
    --good-soft: #14301a;
    --bad: #ff7b72;
    --bad-soft: #3a1b1a;
    --warn: #f2b84b;
    --warn-soft: #362a12;
    --none: #8a939d;
    --none-soft: #1d232a;
    --s1: #3987e5;
    --s2: #d95926;
    --s3: #199e70;
    --s4: #c98500;
    --grid: #232a31;
    --shadow: none;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0e1114;
  --surface: #161a1f;
  --surface-2: #1d232a;
  --ink: #f2f4f6;
  --ink-2: #b7c0ca;
  --muted: #8a939d;
  --line: #2a3139;
  --line-strong: #3a434d;
  --accent: #4fb3d4;
  --accent-soft: #15303a;
  --good: #3fc43f;
  --good-soft: #14301a;
  --bad: #ff7b72;
  --bad-soft: #3a1b1a;
  --warn: #f2b84b;
  --warn-soft: #362a12;
  --none: #8a939d;
  --none-soft: #1d232a;
  --s1: #3987e5;
  --s2: #d95926;
  --s3: #199e70;
  --s4: #c98500;
  --grid: #232a31;
  --shadow: none;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.55 var(--sans);
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 64px; display: grid; gap: 36px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
a:focus-visible, button:focus-visible, select:focus-visible, input:focus-visible, summary:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px;
}
h1, h2, h3 { margin: 0; text-wrap: balance; }
h1 { font-size: 26px; font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 18px; font-weight: 600; }
.mono { font-family: var(--mono); }
.num { font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); }
.eyebrow { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
header.top { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: flex-end; justify-content: space-between; }
header.top p { margin: 6px 0 0; color: var(--ink-2); max-width: 68ch; }
.meta-line { font-size: 12px; color: var(--muted); }
section { display: grid; gap: 14px; }
.section-head { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; justify-content: space-between; }
.section-head p { margin: 0; color: var(--ink-2); font-size: 13px; max-width: 72ch; }

.hw-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }
.hw-card {
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  box-shadow: var(--shadow); padding: 16px; display: grid; gap: 12px; align-content: start;
}
.hw-name { display: flex; gap: 8px; align-items: center; font-weight: 600; font-size: 15px; }
.swatch { width: 10px; height: 10px; border-radius: 3px; flex: none; }
.hw-sub { font-size: 12px; color: var(--muted); margin-top: 2px; word-break: break-word; }
.bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; gap: 2px; background: var(--surface); }
.bar span { display: block; height: 100%; }
.bar .p { background: var(--good); } .bar .f { background: var(--bad); } .bar .e { background: var(--warn); }
.counts { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px 12px; font-size: 12px; }
.counts dt { color: var(--muted); }
.counts dd { margin: 0; font-family: var(--mono); font-variant-numeric: tabular-nums; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 3px 10px; font-size: 12px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; font-family: var(--mono); overflow-wrap: anywhere; }

.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; box-shadow: var(--shadow); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
thead th { font-size: 12px; font-weight: 600; color: var(--ink-2); background: var(--surface-2); position: sticky; top: 0; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
td.metric { text-align: right; white-space: nowrap; }
td.metric .main { font-family: var(--mono); font-variant-numeric: tabular-nums; font-weight: 500; }
td.metric .sub { display: block; font-family: var(--mono); font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
th.hwcol { text-align: right; }
th.hwcol .swatch { display: inline-block; margin-right: 6px; vertical-align: 1px; }
.opname { font-family: var(--mono); font-weight: 500; white-space: nowrap; }
.opnote { display: block; font-size: 11px; color: var(--muted); font-family: var(--sans); font-weight: 400; }

.controls { display: flex; flex-wrap: wrap; gap: 8px 10px; align-items: center; }
.seg { display: inline-flex; border: 1px solid var(--line-strong); border-radius: 8px; overflow: hidden; background: var(--surface); }
.seg button {
  border: 0; background: transparent; color: var(--ink-2); font: 500 12px var(--sans);
  padding: 6px 11px; cursor: pointer;
}
.seg button + button { border-left: 1px solid var(--line); }
.seg button[aria-pressed="true"] { background: var(--accent-soft); color: var(--accent); }
select, input[type="search"] {
  font: 13px var(--sans); color: var(--ink); background: var(--surface);
  border: 1px solid var(--line-strong); border-radius: 8px; padding: 6px 10px;
}
input[type="search"] { min-width: 0; width: 220px; max-width: 100%; }
label.ctl { display: inline-flex; gap: 6px; align-items: center; font-size: 12px; color: var(--ink-2); }

.notes { margin: 0; padding: 12px 16px; font-size: 12px; color: var(--ink-2); display: grid; gap: 4px; list-style: none; border-top: 1px solid var(--line); }
.notes li::before { content: "·"; color: var(--muted); margin-right: 6px; }

.multiples { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.chart { padding: 14px 14px 8px; display: grid; gap: 6px; position: relative; }
.chart-title { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; font-size: 13px; font-weight: 600; }
.chart-title .last { font-family: var(--mono); font-weight: 500; font-variant-numeric: tabular-nums; }
.chart svg { width: 100%; height: auto; display: block; overflow: visible; }
.chart .empty { font-size: 12px; color: var(--muted); padding: 24px 0; text-align: center; }
.tip {
  position: absolute; pointer-events: none; background: var(--ink); color: var(--bg);
  font-size: 11px; line-height: 1.4; padding: 6px 8px; border-radius: 6px; white-space: nowrap; z-index: 2;
  font-variant-numeric: tabular-nums;
}

.pill {
  display: inline-block; min-width: 52px; text-align: center; font: 600 11px/1 var(--mono);
  padding: 5px 7px; border-radius: 999px; letter-spacing: 0.02em;
}
.pill.PASS { background: var(--good-soft); color: var(--good); }
.pill.FAIL { background: var(--bad-soft); color: var(--bad); }
.pill.ERROR { background: var(--warn-soft); color: var(--warn); }
.pill.NONE { background: var(--none-soft); color: var(--none); }
.matrix td.cell { text-align: center; }
.matrix tbody tr.row { cursor: pointer; }
.matrix tbody tr.row:hover td { background: var(--surface-2); }
.matrix tr.detail td { background: var(--surface-2); font-size: 12px; }
.matrix tr.detail dl { margin: 0; display: grid; grid-template-columns: minmax(120px, auto) 1fr; gap: 6px 14px; }
.matrix tr.detail dt { font-weight: 600; color: var(--ink-2); }
.matrix tr.detail dd { margin: 0; font-family: var(--mono); overflow-wrap: anywhere; color: var(--ink-2); }
.module-tag { font-size: 11px; color: var(--muted); font-family: var(--mono); }
.matrix-foot { padding: 10px 14px; font-size: 12px; color: var(--muted); border-top: 1px solid var(--line); }
footer { font-size: 12px; color: var(--muted); }
@media (max-width: 560px) {
  .wrap { padding-inline: 16px; }
  h1 { font-size: 22px; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header class="top">
    <div>
      <div class="eyebrow">SOTA-PNU / lowlevel-api-test</div>
      <h1>Triton 연산자 테스트 보드</h1>
      <p>CPU, GPU, NPU에서 Triton 연산자의 실행·정확도(기능 테스트)와 처리량(성능 테스트)을 GitHub Actions로 측정한 결과입니다.</p>
    </div>
    <div class="meta-line">
      <div id="built"></div>
      <nav id="nav" aria-label="대시보드 이동"></nav>
    </div>
  </header>

  <section aria-labelledby="h-status">
    <div class="section-head">
      <h2 id="h-status">하드웨어별 최신 결과</h2>
      <p>하드웨어마다 가장 최근 실행 기준입니다. 막대는 기능 테스트의 PASS / FAIL / ERROR 비율입니다.</p>
    </div>
    <div class="hw-grid" id="hw-cards"></div>
  </section>

  <section aria-labelledby="h-perf">
    <div class="section-head">
      <h2 id="h-perf">성능 테스트</h2>
      <div class="controls">
        <div class="seg" role="group" aria-label="표시 지표" id="perf-metric">
          <button type="button" data-metric="primary" aria-pressed="true">대표 지표</button>
          <button type="button" data-metric="ms" aria-pressed="false">ms</button>
          <button type="button" data-metric="gbps" aria-pressed="false">GB/s</button>
          <button type="button" data-metric="ops" aria-pressed="false">FLOPS</button>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="scroll"><table id="perf-table"></table></div>
      <ul class="notes">
        <li>대표 지표는 원소별 연산이 GB/s, <span class="mono">matmul</span>이 FLOPS입니다. 원소별 연산은 메모리 대역폭에 막히기 때문입니다.</li>
        <li>GB/s와 FLOPS는 입출력 tensor 크기와 연산 정의로 센 논리값이며, 실제 메모리 전송량이나 명령어 수는 아닙니다.</li>
        <li>입력 크기: CPU/GPU <span class="mono">4096×4096</span> fp32, NPU <span class="mono">1×2048×1024</span> fp32. matmul은 GPU 4096³(TF32), CPU 2048³, NPU 8192×256×1024.</li>
        <li>NPU는 CPU tensor를 입력으로 compiled model 호출 전체를 측정하므로 host↔NPU 전송 시간이 포함됩니다.</li>
      </ul>
    </div>
  </section>

  <section aria-labelledby="h-trend">
    <div class="section-head">
      <h2 id="h-trend">성능 추이</h2>
      <div class="controls">
        <label class="ctl" for="trend-op">연산
          <select id="trend-op"></select>
        </label>
        <label class="ctl" for="trend-metric">지표
          <select id="trend-metric">
            <option value="primary">대표 지표</option>
            <option value="ms">ms (낮을수록 좋음)</option>
            <option value="gbps">GB/s</option>
            <option value="ops">FLOPS</option>
          </select>
        </label>
      </div>
    </div>
    <div class="multiples" id="trend"></div>
  </section>

  <section aria-labelledby="h-func">
    <div class="section-head">
      <h2 id="h-func">기능 테스트 매트릭스</h2>
      <div class="controls">
        <input type="search" id="func-search" placeholder="연산자 이름 검색" aria-label="연산자 이름 검색">
        <label class="ctl" for="func-filter">보기
          <select id="func-filter">
            <option value="all">전체</option>
            <option value="bad">FAIL 또는 ERROR 포함</option>
            <option value="diff">하드웨어마다 결과가 다른 항목</option>
          </select>
        </label>
        <label class="ctl" for="func-module">모듈
          <select id="func-module"><option value="">전체</option></select>
        </label>
      </div>
    </div>
    <div class="panel">
      <div class="scroll"><table class="matrix" id="func-table"></table></div>
      <div class="matrix-foot" id="func-foot"></div>
    </div>
  </section>

  <footer id="footer"></footer>
</div>

<script>
const DATA = __DATA__;
const SERIES = ["--s1", "--s2", "--s3", "--s4"];
const HW = DATA.hardware;
const color = i => `var(${SERIES[i % SERIES.length]})`;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmtRate = (v, unit = "FLOPS") => {
  if (v == null) return "–";
  for (const [s, p] of [[1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "K"]]) if (v >= s) return `${(v / s).toPrecision(4).replace(/\.?0+$/, "")} ${p}${unit}`;
  return `${v.toPrecision(4)} ${unit}`;
};
const fmtMs = v => v == null ? "–" : `${v < 0.1 ? v.toFixed(4) : v < 10 ? v.toFixed(3) : v.toFixed(1)} ms`;
const fmtGbps = v => v == null ? "–" : `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)} GB/s`;
const fmtDate = iso => {
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString("ko-KR", {year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"});
};
const repoUrl = run => run && run.repository ? `https://github.com/${run.repository}` : null;
const commitLink = run => {
  if (!run || !run.sha) return "–";
  const base = repoUrl(run);
  const short = esc(run.sha.slice(0, 7));
  return base ? `<a class="mono" href="${base}/commit/${esc(run.sha)}">${short}</a>` : `<span class="mono">${short}</span>`;
};
const runLink = run => {
  const base = repoUrl(run);
  if (!base || !run.run_id) return "–";
  return `<a href="${base}/actions/runs/${esc(run.run_id)}">${esc(run.workflow || "run")} #${esc(run.run_id)}</a>`;
};
const PRIMARY = op => op === "perf.matmul" ? "ops" : "gbps";
const METRIC = {
  ms: {get: p => p.ms, fmt: fmtMs, label: "ms"},
  gbps: {get: p => p.gbps, fmt: fmtGbps, label: "GB/s"},
  ops: {get: p => p.ops_per_s, fmt: v => fmtRate(v), label: "FLOPS"},
};
const metricFor = (key, op) => METRIC[key === "primary" ? PRIMARY(op) : key];

document.getElementById("built").innerHTML =
  `페이지 생성 ${esc(fmtDate(DATA.built_at))} · 하드웨어 ${HW.length}종`;
if (DATA.site) {
  const root = DATA.site.root.replace(/\/?$/, "/");
  document.getElementById("nav").innerHTML =
    `브랜치 <span class="mono">${esc(DATA.site.branch)}</span> · <a href="${esc(root)}">main 대시보드</a> · <a href="${esc(root)}branches/">브랜치 목록</a>`;
}

// Hardware cards
document.getElementById("hw-cards").innerHTML = HW.map((h, i) => {
  const s = h.latest.summary, f = s.functional, p = s.perf, run = h.latest.run, v = h.latest.versions;
  const pct = n => f.total ? (n / f.total * 100) : 0;
  const versions = Object.entries(v).filter(([, x]) => x).map(([k, x]) => `<dt>${esc(k)}</dt><dd>${esc(x)}</dd>`).join("");
  return `<article class="hw-card">
    <div>
      <div class="hw-name"><span class="swatch" style="background:${color(i)}"></span>${esc(h.hardware.name)}</div>
      <div class="hw-sub">${esc(h.hardware.label)}</div>
    </div>
    <div class="bar" role="img" aria-label="기능 테스트 PASS ${f.pass}, FAIL ${f.fail}, ERROR ${f.error}">
      ${f.pass ? `<span class="p" style="flex:${pct(f.pass)}"></span>` : ""}${f.fail ? `<span class="f" style="flex:${pct(f.fail)}"></span>` : ""}${f.error ? `<span class="e" style="flex:${pct(f.error)}"></span>` : ""}
    </div>
    <dl class="counts">
      <div><dt>기능 테스트</dt><dd>${f.pass} / ${f.total} PASS</dd></div>
      <div><dt>FAIL · ERROR</dt><dd>${f.fail} · ${f.error}</dd></div>
      <div><dt>성능 테스트</dt><dd>${p.pass} / ${p.total} PASS</dd></div>
      <div><dt>실행 기록</dt><dd>${h.history.length}회</dd></div>
    </dl>
    <dl class="kv">
      <dt>측정</dt><dd>${esc(fmtDate(h.latest.generated_at))}</dd>
      <dt>커밋</dt><dd>${commitLink(run)}${run.ref ? ` <span class="muted">(${esc(run.ref)})</span>` : ""}</dd>
      <dt>실행</dt><dd>${runLink(run)}</dd>
      ${run.runner ? `<dt>runner</dt><dd>${esc(run.runner)}</dd>` : ""}
      ${versions}
    </dl>
  </article>`;
}).join("");

// Performance table
const perfOps = [...new Set(HW.flatMap(h => h.latest.results.filter(r => r.module === "perf").map(r => r.name)))];
const OP_NOTES = {
  "perf.copy": "load/store만", "perf.exp": "단항", "perf.add": "이항",
  "perf.sum": "exp(x) / 행 합", "perf.softmax": "행 softmax", "perf.matmul": "계산 중심",
};
let perfMetric = "primary";
function renderPerf() {
  const head = `<thead><tr><th>연산</th>${HW.map((h, i) => `<th class="hwcol"><span class="swatch" style="background:${color(i)}"></span>${esc(h.hardware.name)}</th>`).join("")}</tr></thead>`;
  const rows = perfOps.map(op => {
    const cells = HW.map(h => {
      const r = h.latest.results.find(x => x.name === op);
      if (!r) return `<td class="metric"><span class="muted">–</span></td>`;
      if (r.result !== "PASS") return `<td class="metric"><span class="pill ${esc(r.result)}" title="${esc(r.detail)}">${esc(r.result)}</span></td>`;
      const m = metricFor(perfMetric, op);
      const sub = perfMetric === "ms" ? fmtGbps(r.gbps) : fmtMs(r.ms);
      return `<td class="metric"><span class="main">${m.fmt(m.get(r))}</span><span class="sub">${sub}</span></td>`;
    }).join("");
    return `<tr><td><span class="opname">${esc(op.replace("perf.", ""))}</span><span class="opnote">${esc(OP_NOTES[op] || "")}</span></td>${cells}</tr>`;
  }).join("");
  document.getElementById("perf-table").innerHTML = head + `<tbody>${rows}</tbody>`;
}
document.getElementById("perf-metric").addEventListener("click", e => {
  const b = e.target.closest("button[data-metric]");
  if (!b) return;
  perfMetric = b.dataset.metric;
  for (const x of e.currentTarget.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x === b));
  renderPerf();
});
renderPerf();

// Trends: one small chart per hardware, each on its own linear scale.
const trendOp = document.getElementById("trend-op");
const trendMetric = document.getElementById("trend-metric");
trendOp.innerHTML = perfOps.map(op => `<option value="${esc(op)}">${esc(op.replace("perf.", ""))}</option>`).join("");
if (perfOps.includes("perf.matmul")) trendOp.value = "perf.matmul";

function niceTicks(lo, hi, n = 4) {
  if (hi <= lo) hi = lo + (lo === 0 ? 1 : Math.abs(lo) * 0.1);
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(s => s >= raw);
  const start = Math.floor(lo / step) * step, ticks = [];
  for (let t = start; t <= hi + step * 1e-9; t += step) ticks.push(+t.toPrecision(12));
  if (ticks[ticks.length - 1] < hi) ticks.push(+(ticks[ticks.length - 1] + step).toPrecision(12));
  return ticks;
}
function renderTrend() {
  const op = trendOp.value, mkey = trendMetric.value, m = metricFor(mkey, op);
  const W = 320, H = 150, L = 52, R = 10, T = 10, B = 24;
  document.getElementById("trend").innerHTML = HW.map((h, i) => {
    const pts = h.history.map(e => ({e, v: e.perf[op] && e.perf[op].result === "PASS" ? m.get(e.perf[op]) : null})).filter(p => p.v != null);
    const last = pts.length ? m.fmt(pts[pts.length - 1].v) : "–";
    let body = `<div class="empty">측정값 없음</div>`;
    if (pts.length) {
      const vals = pts.map(p => p.v), ticks = niceTicks(0, Math.max(...vals));
      const ymax = ticks[ticks.length - 1];
      const x = k => pts.length === 1 ? L + (W - L - R) / 2 : L + k * (W - L - R) / (pts.length - 1);
      const y = v => T + (H - T - B) * (1 - v / ymax);
      const grid = ticks.map(t => `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="var(--grid)" stroke-width="1"/><text x="${L - 6}" y="${y(t) + 3.5}" text-anchor="end" font-size="10" fill="var(--muted)" font-family="var(--mono)">${esc(m.fmt(t).replace(/ (ms|GB\/s)$/, ""))}</text>`).join("");
      const line = pts.map((p, k) => `${k ? "L" : "M"}${x(k).toFixed(1)},${y(p.v).toFixed(1)}`).join("");
      const area = pts.length > 1 ? `<path d="${line}L${x(pts.length - 1).toFixed(1)},${y(0)}L${x(0).toFixed(1)},${y(0)}Z" fill="${color(i)}" opacity="0.10"/>` : "";
      const dots = pts.map((p, k) => `<circle cx="${x(k)}" cy="${y(p.v)}" r="${k === pts.length - 1 ? 4 : 2.5}" fill="${color(i)}" stroke="var(--surface)" stroke-width="2"/>`).join("");
      const first = new Date(pts[0].e.generated_at), lastD = new Date(pts[pts.length - 1].e.generated_at);
      const dl = d => isNaN(d) ? "" : `${d.getMonth() + 1}/${d.getDate()}`;
      const axis = `<line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" stroke="var(--line-strong)"/><text x="${L}" y="${H - 6}" font-size="10" fill="var(--muted)">${dl(first)}</text>${pts.length > 1 ? `<text x="${W - R}" y="${H - 6}" text-anchor="end" font-size="10" fill="var(--muted)">${dl(lastD)}</text>` : ""}`;
      const hits = pts.map((p, k) => {
        const half = pts.length === 1 ? (W - L - R) / 2 : (W - L - R) / (pts.length - 1) / 2;
        return `<rect x="${x(k) - half}" y="${T}" width="${half * 2}" height="${H - T - B}" fill="transparent" data-k="${k}"/>`;
      }).join("");
      body = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(h.hardware.name)} ${esc(op)} ${esc(m.label)} 추이">${grid}${axis}${area}<path d="${line}" fill="none" stroke="${color(i)}" stroke-width="2" stroke-linejoin="round"/>${dots}<g class="hits">${hits}</g></svg>`;
    }
    return `<div class="panel chart" data-hw="${i}"><div class="chart-title"><span><span class="swatch" style="display:inline-block;background:${color(i)};margin-right:6px"></span>${esc(h.hardware.name)}</span><span class="last">${esc(last)}</span></div>${body}</div>`;
  }).join("");
  for (const panel of document.querySelectorAll("#trend .chart")) {
    const h = HW[+panel.dataset.hw];
    const pts = h.history.filter(e => e.perf[op] && e.perf[op].result === "PASS" && m.get(e.perf[op]) != null);
    let tip = null;
    panel.addEventListener("mousemove", ev => {
      const r = ev.target.closest("rect[data-k]");
      if (!r) { tip && tip.remove(); tip = null; return; }
      const e = pts[+r.dataset.k];
      if (!tip) { tip = document.createElement("div"); tip.className = "tip"; panel.appendChild(tip); }
      tip.innerHTML = `${esc(m.fmt(m.get(e.perf[op])))}<br>${esc(fmtDate(e.generated_at))}<br>${e.run.sha ? esc(e.run.sha.slice(0, 7)) : ""} ${e.run.ref ? esc(e.run.ref) : ""}`;
      const box = panel.getBoundingClientRect();
      const left = Math.min(ev.clientX - box.left + 12, box.width - tip.offsetWidth - 4);
      tip.style.left = `${Math.max(4, left)}px`;
      tip.style.top = `${ev.clientY - box.top - tip.offsetHeight - 10}px`;
    });
    panel.addEventListener("mouseleave", () => { tip && tip.remove(); tip = null; });
  }
}
trendOp.addEventListener("change", renderTrend);
trendMetric.addEventListener("change", renderTrend);
renderTrend();

// Functional matrix
const funcRows = new Map();
HW.forEach((h, i) => {
  for (const r of h.latest.results) {
    if (r.module === "perf") continue;
    if (!funcRows.has(r.name)) funcRows.set(r.name, {name: r.name, module: r.module, cells: []});
    funcRows.get(r.name).cells[i] = r;
  }
});
const rowsAll = [...funcRows.values()].sort((a, b) => a.name.localeCompare(b.name));
const modSel = document.getElementById("func-module");
modSel.innerHTML += [...new Set(rowsAll.map(r => r.module))].sort().map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join("");
const openRows = new Set();
function renderFunc() {
  const q = document.getElementById("func-search").value.trim().toLowerCase();
  const mode = document.getElementById("func-filter").value, mod = modSel.value;
  const status = (row, i) => row.cells[i] ? row.cells[i].result : "NONE";
  const rows = rowsAll.filter(row => {
    if (q && !row.name.toLowerCase().includes(q)) return false;
    if (mod && row.module !== mod) return false;
    const st = HW.map((_, i) => status(row, i));
    if (mode === "bad") return st.some(s => s === "FAIL" || s === "ERROR");
    if (mode === "diff") return new Set(st.filter(s => s !== "NONE")).size > 1;
    return true;
  });
  const head = `<thead><tr><th>연산자</th>${HW.map((h, i) => `<th style="text-align:center"><span class="swatch" style="display:inline-block;background:${color(i)};margin-right:6px"></span>${esc(h.hardware.name)}</th>`).join("")}</tr></thead>`;
  const body = rows.map(row => {
    const cells = HW.map((_, i) => {
      const st = status(row, i), r = row.cells[i];
      return `<td class="cell"><span class="pill ${st}"${r ? ` title="${esc(r.detail)}"` : ""}>${st === "NONE" ? "–" : st}</span></td>`;
    }).join("");
    const open = openRows.has(row.name);
    const detail = open ? `<tr class="detail"><td colspan="${HW.length + 1}"><dl>${HW.map((h, i) => {
      const r = row.cells[i];
      return `<dt>${esc(h.hardware.name)}</dt><dd>${r ? `${esc(r.result)} · ${esc(r.dtype)} · ${esc(r.detail) || "–"}` : "실행하지 않음"}</dd>`;
    }).join("")}</dl></td></tr>` : "";
    return `<tr class="row" data-name="${esc(row.name)}" tabindex="0" aria-expanded="${open}"><td><span class="opname">${esc(row.name)}</span> <span class="module-tag">${esc(row.module)}</span></td>${cells}</tr>${detail}`;
  }).join("");
  document.getElementById("func-table").innerHTML = head + `<tbody>${body || `<tr><td colspan="${HW.length + 1}" class="muted">조건에 맞는 항목이 없습니다.</td></tr>`}</tbody>`;
  document.getElementById("func-foot").textContent = `${rows.length} / ${rowsAll.length}개 항목 · 행을 누르면 하드웨어별 상세 메시지가 열립니다.`;
}
const toggleRow = tr => { const n = tr.dataset.name; openRows.has(n) ? openRows.delete(n) : openRows.add(n); renderFunc(); };
document.getElementById("func-table").addEventListener("click", e => { const tr = e.target.closest("tr.row"); if (tr) toggleRow(tr); });
document.getElementById("func-table").addEventListener("keydown", e => {
  const tr = e.target.closest("tr.row");
  if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggleRow(tr); }
});
for (const id of ["func-search", "func-filter", "func-module"]) document.getElementById(id).addEventListener("input", renderFunc);
renderFunc();

document.getElementById("footer").innerHTML =
  `데이터는 각 실행의 JSON 보고서(<span class="mono">triton_test.py --json-out</span>)에서 만들어집니다. 소스: <a href="https://github.com/SOTA-PNU/lowlevel-api-test">SOTA-PNU/lowlevel-api-test</a>`;
</script>
"""

INDEX_BODY = r"""
<div class="wrap">
  <header class="top">
    <div>
      <div class="eyebrow">SOTA-PNU / lowlevel-api-test</div>
      <h1>브랜치별 테스트 보드</h1>
      <p>브랜치마다 가장 최근 CI 실행 결과로 만든 대시보드입니다. <a href="../">main 대시보드</a>가 기준 결과입니다.</p>
    </div>
  </header>
  <section>
    <div class="panel"><div class="scroll"><table>
      <thead><tr><th>브랜치</th><th>하드웨어</th><th>최근 커밋</th><th>페이지 생성</th></tr></thead>
      <tbody>__ROWS__</tbody>
    </table></div></div>
  </section>
</div>
"""

def _wrap_page(page):
    head, body = page.split('\n<div class="wrap">', 1)
    return ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f'{head}\n</head>\n<body>\n<div class="wrap">{body}\n</body>\n</html>\n')

def write_dashboard(data_dir, out_dir, site=None, fragment=False):
    payload = build_payload(load_reports(data_dir), site)
    if not payload["hardware"]:
        raise SystemExit(f"no reports found under {data_dir}")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.replace("__DATA__", data)
    (out / "index.html").write_text(page if fragment else _wrap_page(page))
    (out / "data.json").write_text(json.dumps(payload, ensure_ascii=False))
    names = ", ".join(h["hardware"]["name"] for h in payload["hardware"])
    print(f"wrote {out / 'index.html'} ({len(payload['hardware'])} hardware: {names})")
    return payload

def write_branch_index(branches_dir):
    """List every branches/<slug>/data.json dashboard in branches/index.html."""
    branches_dir = pathlib.Path(branches_dir)
    rows = []
    for data_path in sorted(branches_dir.glob("*/data.json")):
        payload = json.loads(data_path.read_text())
        site = payload.get("site") or {}
        runs = [h["latest"]["run"] for h in payload["hardware"]]
        latest = max(runs, key=lambda r: r.get("run_id") or "", default={})
        sha = latest.get("sha") or ""
        commit = (f'<a class="mono" href="https://github.com/{html.escape(latest["repository"])}/commit/{html.escape(sha)}">{html.escape(sha[:7])}</a>'
                  if sha and latest.get("repository") else "–")
        hardware = ", ".join(html.escape(h["hardware"]["name"]) for h in payload["hardware"])
        rows.append(
            f'<tr><td><a class="mono" href="{html.escape(data_path.parent.name)}/">{html.escape(site.get("branch", data_path.parent.name))}</a></td>'
            f'<td>{hardware}</td><td>{commit}</td><td class="num">{html.escape(payload["built_at"])}</td></tr>'
        )
    body = "".join(rows) or '<tr><td colspan="4" class="muted">게시된 브랜치가 없습니다.</td></tr>'
    head = TEMPLATE.split('\n<div class="wrap">', 1)[0].replace(
        "<title>Triton 연산자 테스트 보드</title>", "<title>Triton 테스트 브랜치 목록</title>")
    branches_dir.mkdir(parents=True, exist_ok=True)
    (branches_dir / "index.html").write_text(_wrap_page(head + INDEX_BODY.replace("__ROWS__", body)))
    print(f"wrote {branches_dir / 'index.html'} ({len(rows)} branches)")

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Directory searched recursively for *.json reports")
    parser.add_argument("--out", required=True, help="Output directory for index.html and data.json")
    parser.add_argument("--fragment", action="store_true", help="Write the page body only (no <html> skeleton)")
    args = parser.parse_args()
    write_dashboard(args.data, args.out, fragment=args.fragment)

if __name__ == "__main__":
    main()
