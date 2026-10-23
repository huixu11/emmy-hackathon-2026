#!/usr/bin/env python3
"""Kernel IR -> PTX.

    python compile.py path/to/kernel.ir.json > out.ptx

PTX goes to stdout; diagnostics go to stderr with a nonzero exit code.
"""

import sys

from ptxc import CompileError, compile_file


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json > out.ptx", file=sys.stderr)
        return 2
    try:
        ptx = compile_file(argv[1])
    except CompileError as exc:
        print(f"compile.py: {argv[1]}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:  # a missing or unreadable IR file
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
