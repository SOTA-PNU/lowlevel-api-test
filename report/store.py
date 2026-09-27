"""Copy downloaded JSON reports into the results store.

Usage: python3 report/store.py SRC_DIR DATA_DIR

Each report lands at DATA_DIR/<hardware id>/<run id>-<attempt>.json, so a
report that is collected twice overwrites itself instead of duplicating.
Standard library only.
"""
import json
import pathlib
import sys

def main(src, dst):
    stored = 0
    for path in sorted(pathlib.Path(src).rglob("*.json")):
        try:
            report = json.loads(path.read_text())
            hw_id = report["hardware"]["id"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"skip {path}: {exc}")
            continue
        run = report.get("run") or {}
        run_key = f"{run.get('run_id')}-{run.get('run_attempt') or 1}" if run.get("run_id") else report["generated_at"]
        target = pathlib.Path(dst) / hw_id / f"{run_key}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        stored += 1
        print(f"stored {target}")
    print(f"{stored} report(s) stored")

if __name__ == "__main__":
    main(*sys.argv[1:3])
