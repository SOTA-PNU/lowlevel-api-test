"""Update a gh-pages checkout with one branch's test results.

Usage:
  python3 report/publish.py --site DIR --branch NAME --incoming DIR [--default-branch main]
  python3 report/publish.py --site DIR --branch NAME --remove

Site layout (served by GitHub Pages from the gh-pages branch):
  index.html, data/<hw>/<run>.json                 default branch
  branches/<slug>/index.html, branches/<slug>/data  every other branch
  branches/index.html                               list of branch dashboards
Standard library only.
"""
import argparse
import pathlib
import re
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_site  # noqa: E402
import store  # noqa: E402

def branch_slug(branch):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip("-") or "branch"

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True, help="gh-pages checkout")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--default-branch", default="main")
    parser.add_argument("--incoming", help="Directory with downloaded results-* artifacts")
    parser.add_argument("--remove", action="store_true", help="Delete the branch dashboard")
    args = parser.parse_args()

    site = pathlib.Path(args.site)
    site.mkdir(parents=True, exist_ok=True)
    (site / ".nojekyll").touch()
    is_default = args.branch == args.default_branch
    target = site if is_default else site / "branches" / branch_slug(args.branch)

    if args.remove:
        if is_default:
            raise SystemExit("refusing to remove the default branch dashboard")
        shutil.rmtree(target, ignore_errors=True)
        print(f"removed {target}")
    else:
        if not args.incoming:
            raise SystemExit("--incoming is required unless --remove is given")
        store.main(args.incoming, target / "data")
        root = "./" if is_default else "../../"
        build_site.write_dashboard(target / "data", target, site={"branch": args.branch, "root": root})
    build_site.write_branch_index(site / "branches")

if __name__ == "__main__":
    main()
