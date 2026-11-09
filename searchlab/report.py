from __future__ import annotations

import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .common import save


def stats(values, digits=3):
    if not values:
        return "unavailable"
    if any(v is None for v in values):
        return "unknown"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values) if len(values) > 1 else 0:.{digits}f}"


def fixture_report(directory, results):
    """Software coverage only: never publish a fake method leaderboard."""
    title = "OFFLINE SOFTWARE CHECK — NO MODEL EVALUATION"
    text = [f"# {title}", "", "No real models were evaluated. Model replies and search proposals were scripted.",
            "This report checks software execution only. It cannot establish whether AFlow or ADAS performs better.",
            "Model-performance scores, rankings, gains over baseline and learning curves are unavailable in offline mode.",
            "", "| Task family | Completed test runs | Incomplete / not run | Recorded task errors |", "|---|---:|---:|---:|"]
    for benchmark in sorted({r["benchmark"] for r in results}):
        rows = [r for r in results if r["benchmark"] == benchmark]
        complete = [r for r in rows if r["test"]["status"] == "complete"]
        errors = sum(r["test"]["result"]["errors"] for r in complete)
        text.append(f"| {benchmark} | {len(complete)}/{len(rows)} | {len(rows) - len(complete)} | {errors} |")
    controls = set()
    for path in directory.glob("*/splits.json"):
        bundle = json.loads(path.read_text())
        controls.update(row.get("fixture_outcome") for row in bundle.get("test", []))
    if {"correct", "wrong", "error"} <= controls:
        text.extend(["", "## Controlled cases", "", "Test inputs explicitly request correct replies, wrong replies and deliberate failures from the stub.",
                     "These cases verify that wrong answers and failing programs do not receive full credit. Recorded task errors include intentional failures and incorrect programs that fail their tests.",
                     "Raw task scores are retained in JSON and the check_score CSV column for debugging only; they are not measurements of agent capability."])
    else:
        text.extend(["", "## Legacy sample", "", "This earlier sample scripted correct answers only, which explains its perfect raw scores. It did not test mixed success/failure cases.",
                     "Run the current offline configuration for controlled wrong-answer and execution-failure coverage."])
    text.extend(["", "## Real comparisons", "", "Configure a real model endpoint, credentials and spending limits, then run with --live. Actual benchmark performance is not yet measured."])
    markdown = "\n".join(text) + "\n"
    (directory / "report.md").write_text(markdown)
    (directory / "report.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#172638}}pre{{white-space:pre-wrap;line-height:1.6}}.notice{{background:#fff1c5;padding:18px;border-radius:8px}}</style>
<h1>Offline software check</h1><p class="notice">No real model evaluation has been run. Agent performance is unmeasured.</p><pre>{html.escape(markdown)}</pre></html>''')
    return directory / "report.html"


def report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    groups = defaultdict(list)
    flat = []
    for row in results:
        groups[(row["benchmark"], row["method"])].append(row)
        test = row["test"].get("result", {})
        flat.append({"benchmark": row["benchmark"], "method": row["method"], "seed": row["seed"],
                     "result_kind": "software_check" if manifest["fixture"] else "model_evaluation",
                     "test_status": row["test"]["status"], "check_score" if manifest["fixture"] else "test_score": test.get("score"),
                     "search_status": row["search"]["status"], "search_usd": row["search"]["budget"]["usd"],
                     "search_calls": row["search"]["budget"]["calls"],
                     "completed_generated_candidates": row["search"].get("completed_generated_candidates"),
                     "selected_origin": row.get("selected_origin"),
                     "search_tokens": row["search"]["budget"].get("tokens"),
                     "search_token_usage_complete": row["search"]["budget"].get("token_usage_complete", True),
                     "search_input_tokens": row["search"]["budget"].get("input_tokens"),
                     "search_output_tokens": row["search"]["budget"].get("output_tokens"),
                     "search_cached_input_tokens": row["search"]["budget"].get("cached_input_tokens"),
                     "search_seconds": row["search"]["budget"].get("seconds"),
                     "search_codex_runs": row["search"]["budget"].get("codex_runs"),
                     "search_failures": row["search"]["budget"].get("failures"),
                     "search_cost_uncertain": row["search"]["budget"]["cost_uncertain"],
                     "test_cost_uncertain": row["test"].get("budget", {}).get("cost_uncertain"),
                     "test_usd": row["test"].get("budget", {}).get("usd"),
                     "test_tokens": row["test"].get("budget", {}).get("tokens"),
                     "test_token_usage_complete": row["test"].get("budget", {}).get("token_usage_complete"),
                     "test_input_tokens": row["test"].get("budget", {}).get("input_tokens"),
                     "test_output_tokens": row["test"].get("budget", {}).get("output_tokens"),
                     "test_cached_input_tokens": row["test"].get("budget", {}).get("cached_input_tokens"),
                     "test_seconds": row["test"].get("budget", {}).get("seconds"),
                     "test_codex_runs": row["test"].get("budget", {}).get("codex_runs"),
                     "test_errors": test.get("errors"), "test_count": test.get("count"),
                     "attack_success": test.get("attack_success"), "clean_utility": test.get("clean_utility"),
                     "attacked_utility": test.get("attacked_utility")})
    with (directory / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    if manifest["fixture"]:
        return fixture_report(directory, results)
    title = "OFFLINE FIXTURE CHECK — NOT BENCHMARK RESULTS" if manifest["fixture"] else "AFlow and ADAS benchmark comparison"
    text = [f"# {title}", "", "Scores are per benchmark, mean ± sample standard deviation across seeds. Higher is better; attack success is lower-is-better.",
            "Incomplete test runs have no score. Search costs include proposal, initial archive, debugging and candidate evaluation.",
            "", "| Benchmark | Method | Complete seeds | Test score | New candidates | Search USD | Test USD/task | Errors |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    if manifest.get("total_budget"):
        total = manifest["total_budget"]
        text[2:2] = [f"Whole-run measured usage: {total['tokens']:,} tokens / {total['limits']['max_tokens']:,} configured limit; {total['calls']} gateway calls; {total['seconds']:.1f} seconds. Token usage complete: {total['token_usage_complete']}.", ""]
    for (benchmark, method), rows in sorted(groups.items()):
        complete = [r for r in rows if r["test"]["status"] == "complete"]
        text.append(f"| {benchmark} | {method} | {len(complete)}/{len(rows)} | {stats([r['test']['result']['score'] for r in complete])} | "
                    f"{sum(r['search'].get('completed_generated_candidates', 0) for r in rows) if all('completed_generated_candidates' in r['search'] for r in rows) else 'unknown'} | "
                    f"{stats([r['search']['budget']['usd'] for r in rows], 6)} | "
                    f"{stats([r['test']['result']['usd']/r['test']['result']['count'] if r['test']['result']['usd'] is not None else None for r in complete], 6)} | "
                    f"{sum(r['test']['result']['errors'] for r in complete)} |")
    text.extend(["", "## Measured usage", "", "Calls count gateway requests; with Codex, each request launches a fresh CLI session. Dollar cost for ChatGPT-authenticated Codex is unknown, not zero.",
                 "", "| Benchmark | Method | Search calls | Search tokens | Search seconds | Test tokens/task | Test seconds/task |", "|---|---|---:|---:|---:|---:|---:|"])
    for (benchmark, method), rows in sorted(groups.items()):
        complete = [r for r in rows if r["test"]["status"] == "complete"]
        text.append(f"| {benchmark} | {method} | {stats([r['search']['budget']['calls'] for r in rows], 1)} | "
                    f"{stats([r['search']['budget']['tokens'] if r['search']['budget'].get('token_usage_complete', True) else None for r in rows], 0)} | "
                    f"{stats([r['search']['budget']['seconds'] for r in rows], 1)} | "
                    f"{stats([r['test']['result'].get('tokens', r['test']['budget']['tokens'])/r['test']['result']['count'] for r in complete], 1)} | "
                    f"{stats([r['test']['result'].get('seconds', r['test']['budget']['seconds'])/r['test']['result']['count'] for r in complete], 2)} |")
    text.extend(["", "## Paired improvement over baseline", "", "Use identical seeds and task splits. These differences measure improvement within a benchmark; they do not combine unlike metrics.", ""])
    for (benchmark, method), rows in sorted(groups.items()):
        if method == "baseline":
            continue
        baseline = {r["seed"]: r for r in groups[(benchmark, "baseline")] if r["test"]["status"] == "complete"}
        pairs = [(r, baseline[r["seed"]]) for r in rows if r["test"]["status"] == "complete" and r["seed"] in baseline]
        delta = [r["test"]["result"]["score"] - b["test"]["result"]["score"] for r, b in pairs]
        text.append(f"- {benchmark}, {method}: {stats(delta)} ({len(delta)} paired seeds).")
    text.extend(["", "## Run status", ""])
    text.append("New candidates counts only generated agents with a complete validation evaluation. Zero means the run did not validate a new design; its selected agent can come from the initial archive. Such a run does not demonstrate improvement from automated search.")
    for row in results:
        if row["search"]["status"] != "complete" or row["test"]["status"] != "complete":
            text.append(f"- {row['benchmark']}/{row['method']}/seed {row['seed']}: search {row['search']['status']} ({row['search']['error']}), test {row['test']['status']}.")
    text.extend(["", "## Interpretation", "", "AFlow uses its upstream workflow-search step and tree experience. ADAS uses its upstream archive search, initial architectures and two reflection steps.",
                 "Both use a common restricted execution interface and gateway; this is an adapted comparison, not a reproduction of the papers' original benchmark results.",
                 "The same search caps apply to both methods; an iteration limit can stop a method before it spends its entire budget. API seeds are best effort.",
                 "Dollar totals use the configured rates, including a conservative full-price treatment of cached input. They are estimates, not provider invoices."])
    markdown = "\n".join(text) + "\n"
    (directory / "report.md").write_text(markdown)
    # Self-contained offline report with per-seed validation learning curves.
    curves = []
    for benchmark in sorted({r["benchmark"] for r in results}):
        series = []
        for row in results:
            if row["benchmark"] != benchmark or row["method"] == "baseline":
                continue
            path = directory / benchmark / f"seed_{row['seed']}" / row["method"] / "history.json"
            if path.exists():
                history = json.loads(path.read_text())
                best = 0
                points = []
                for entry in history:
                    best = max(best, entry["result"]["score"])
                    points.append((entry["budget"]["calls"], best))
                series.append((row["method"], row["seed"], points))
        max_x = max([x for _, _, points in series for x, _ in points] or [1])
        lines = []
        for method, seed, points in series:
            color = "#2073c4" if method == "aflow" else "#c45520"
            coordinates = " ".join(f"{55+620*x/max_x:.1f},{205-175*y:.1f}" for x, y in points)
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" opacity="0.65" points="{coordinates}"><title>{method}, seed {seed}</title></polyline>')
        curves.append(f'<h2>{html.escape(benchmark)}</h2><p>Best completed validation score vs. total search model calls; blue: AFlow, orange: ADAS. One line per seed.</p><svg viewBox="0 0 720 245" role="img" aria-label="Validation learning curves"><path d="M55 25 V205 H675" fill="none" stroke="#777"/><text x="20" y="35">1.0</text><text x="20" y="210">0.0</text><text x="55" y="230">0 calls</text><text x="620" y="230">{max_x} calls</text>{"".join(lines)}</svg>')
    rows_html = "".join("<tr>" + "".join(f"<td>{html.escape(str(r[k]) if r[k] is not None else '—')}</td>" for k in flat[0]) + "</tr>" for r in flat)
    headers = "".join(f"<th>{html.escape(k)}</th>" for k in flat[0])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#172638;background:#f7f9fc}}h1{{font-size:28px}}pre{{white-space:pre-wrap;line-height:1.5}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:9px;border:1px solid #ccd4df;text-align:left}}.scroll{{overflow:auto}}svg{{max-width:720px;background:white}}.notice{{background:#fff1c5;padding:16px;border-radius:8px}}</style>
<h1>{html.escape(title)}</h1><p class="notice">{"Synthetic fixture provider: these numbers verify software plumbing only." if manifest['fixture'] else "Adapted upstream methods, shared gateway and held-out task partitions. Review run status before drawing conclusions."}</p>
<pre>{html.escape(markdown)}</pre>{''.join(curves)}<h2>Every run</h2><div class="scroll"><table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table></div></html>'''
    (directory / "report.html").write_text(document)
    return directory / "report.html"
