"""CLI for ggml-stack: drive a single-ggml build from the command line.

    python -m ggml_stack build [--root DIR] [--engines llama,whisper,sd]
                               [--with-server] [--openssl] [--no-shared-ggml]
                               [--dry-run]
    python -m ggml_stack info  [--root DIR]

`--dry-run` prints the exact git/cmake command lines without executing them,
which is also what the test-suite asserts on.
"""

from __future__ import annotations

import argparse
import sys

from .project import Project
from .recipe import ENGINES, StackConfig, build_stack, stack_summary
from .shell import setup_logging


def _parse_engines(value: str) -> tuple[str, ...]:
    engines = tuple(e.strip() for e in value.split(",") if e.strip())
    bad = [e for e in engines if e not in ENGINES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown engine(s): {bad}; choose from {ENGINES}")
    return engines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ggml_stack", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="fetch + build the engines against one ggml")
    b.add_argument("--root", default=None, help="project root (default: cwd)")
    b.add_argument("--engines", type=_parse_engines, default=ENGINES,
                   help="comma-separated subset of llama,whisper,sd")
    b.add_argument("--with-server", action="store_true",
                   help="build llama-server (server-context + cpp-httplib)")
    b.add_argument("--openssl", action="store_true", help="LLAMA_OPENSSL=ON")
    b.add_argument("--no-shared-ggml", action="store_true",
                   help="let SD use its own vendored ggml (default: share llama's)")
    b.add_argument("--sd-examples", action="store_true")
    b.add_argument("--dry-run", action="store_true",
                   help="print git/cmake commands without running them")

    i = sub.add_parser("info", help="show staged archives under thirdparty/")
    i.add_argument("--root", default=None)

    args = parser.parse_args(argv)
    setup_logging()

    if args.command == "build":
        cfg = StackConfig(
            root=args.root,
            engines=args.engines,
            shared_ggml=not args.no_shared_ggml,
            with_server=args.with_server,
            openssl=args.openssl,
            sd_examples=args.sd_examples,
            dry_run=args.dry_run,
        )
        project = build_stack(cfg)
        if not args.dry_run:
            for engine, libs in stack_summary(project).items():
                print(f"{engine}: {', '.join(libs) or '(no archives)'}")
        return 0

    if args.command == "info":
        project = Project(root=args.root)
        summary = stack_summary(project)
        if not summary:
            print(f"no staged engines under {project.thirdparty}")
            return 1
        for engine, libs in summary.items():
            print(f"{engine}: {', '.join(libs) or '(no archives)'}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
