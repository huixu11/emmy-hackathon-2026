#!/usr/bin/env python3
"""Kernel IR -> PTX.

    python compile.py path/to/kernel.ir.json > out.ptx
    python compile.py --exact-f64 path/to/kernel.ir.json > out.ptx

PTX goes to stdout; diagnostics go to stderr with a nonzero exit code. f64 values are computed
in f32, as emmy's own CUDA does, unless --exact-f64 (or PTXC_EXACT_F64=1) is given.
"""

import sys

from ptxc import CompileError, compile_file


def main(argv: list[str]) -> int:
    exact = "--exact-f64" in argv[1:]
    args = [a for a in argv[1:] if a != "--exact-f64"]
    if len(args) != 1:
        print("usage: python compile.py [--exact-f64] path/to/kernel.ir.json > out.ptx", file=sys.stderr)
        return 2
    try:
        ptx = compile_file(args[0], exact_f64=True if exact else None)
    except CompileError as exc:
        print(f"compile.py: {args[0]}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:  # a missing or unreadable IR file
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
