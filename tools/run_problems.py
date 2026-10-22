#!/usr/bin/env python3
"""Run submit.py's check (compile + run on the GPU runner + compare) on several problems.

    python tools/run_problems.py                                 # every problem
    python tools/run_problems.py p00_scalar_add r01_array_sum    # just these
    python tools/run_problems.py --dir tests/problems            # every problem in a folder
    python tools/run_problems.py --via-proxy                     # see below

Prints one line per problem, the full submit.py output for failures, and exits 0 only if all
pass. The compiler is this repo's compile.py unless --compiler names another.

--via-proxy sends the runner requests through a CONNECT tunnel of the proxy in $HTTPS_PROXY,
for sandboxes whose proxy refuses plain-HTTP requests (the runner speaks plain HTTP).
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import io
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import submit  # noqa: E402


def tunneled_post(server: str, path: str, body: dict) -> dict:
    """submit.post, sent through the $HTTPS_PROXY CONNECT tunnel."""
    proxy, target = urlparse(os.environ["HTTPS_PROXY"]), urlparse(server)
    conn = http.client.HTTPConnection(proxy.hostname, proxy.port, timeout=300)
    conn.set_tunnel(target.hostname, target.port or 80)
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
    finally:
        conn.close()
    if resp.status >= 400:
        sys.exit(f"runner answered HTTP {resp.status}: {data[:300]!r}")
    return json.loads(data)


def check(problem: str, compiler: str) -> tuple[bool, list[str]]:
    """Run submit.py's main() for one problem; (passed, its output lines)."""
    out = io.StringIO()
    saved_argv = sys.argv
    sys.argv = ["submit.py", problem, "--compiler", compiler]
    try:
        with contextlib.redirect_stdout(out):
            code = submit.main()
    except SystemExit as exc:  # submit.py exits with a message on setup errors
        code = exc.code
        if isinstance(code, str):
            out.write(code + "\n")
    finally:
        sys.argv = saved_argv
    return code == 0, out.getvalue().strip().splitlines()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="*", help="problem ids or folders (default: all of --dir)")
    ap.add_argument("--dir", type=Path, default=submit.PROBLEMS, help="folder of problems (default: problems/)")
    ap.add_argument("--compiler", default=str(REPO / "compile.py"))
    ap.add_argument("--via-proxy", action="store_true", help="tunnel runner requests through $HTTPS_PROXY")
    args = ap.parse_args()

    if args.via_proxy:
        submit.post = tunneled_post
    problems = args.problems or sorted(str(p) for p in args.dir.iterdir() if (p / "meta.json").exists())

    failed = 0
    for problem in problems:
        passed, lines = check(problem, args.compiler)
        print(f"{'PASS' if passed else 'FAIL'}  {Path(problem).name:20s} {lines[-1] if lines else ''}")
        if not passed:
            failed += 1
            print("\n".join("      " + line for line in lines))
    print(f"\n{len(problems) - failed}/{len(problems)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
