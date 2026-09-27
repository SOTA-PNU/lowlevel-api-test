"""Render one or more triton_test.py JSON reports as GitHub Step Summary markdown.

Usage: python3 report/summary.py results.json [...] >> "$GITHUB_STEP_SUMMARY"
Standard library only: it runs on the runner host, not in the test image.
"""
import json
import sys

def format_rate(value, unit):
    if value is None:
        return "-"
    for scale, prefix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if value >= scale:
            return f"{value / scale:.4g} {prefix}{unit}"
    return f"{value:.4g} {unit}"

def format_number(value, digits):
    return "-" if value is None else f"{value:.{digits}f}"

def render(report):
    hw, run, versions, summary = report["hardware"], report["run"], report["versions"], report["summary"]
    lines = [f"### {hw['label']}", ""]
    meta = [f"`{hw['id']}`"]
    if run.get("sha"):
        meta.append(f"commit `{run['sha'][:7]}`")
    if run.get("runner"):
        meta.append(f"runner `{run['runner']}`")
    meta += [f"{name} `{version}`" for name, version in versions.items() if version]
    lines += [" · ".join(meta), ""]

    lines += ["| tests | total | PASS | FAIL | ERROR |", "|---|---:|---:|---:|---:|"]
    for group in ("functional", "perf"):
        counts = summary[group]
        lines.append(f"| {group} | {counts['total']} | {counts['pass']} | {counts['fail']} | {counts['error']} |")
    lines.append("")

    perf = [r for r in report["results"] if r["module"] == "perf"]
    if perf:
        lines += ["| perf | result | ms | GB/s | FLOPS/OPS |", "|---|---|---:|---:|---:|"]
        for r in perf:
            lines.append(
                f"| `{r['name']}` | {r['result']} | {format_number(r['ms'], 4)} | "
                f"{format_number(r['gbps'], 2)} | {format_rate(r['ops_per_s'], r['ops_unit'] or 'FLOPS')} |"
            )
        lines.append("")

    bad = [r for r in report["results"] if r["result"] != "PASS"]
    if bad:
        lines += [f"<details><summary>FAIL/ERROR ({len(bad)})</summary>", ""]
        for r in bad:
            lines.append(f"- `{r['name']}` {r['result']}: {r['detail'][:200]}")
        lines += ["", "</details>", ""]
    return "\n".join(lines)

def main(paths):
    for path in paths:
        with open(path) as file:
            print(render(json.load(file)))

if __name__ == "__main__":
    main(sys.argv[1:])
