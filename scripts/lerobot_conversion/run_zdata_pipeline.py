#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the config-driven zdata_hdf5 to GR00T pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from zdata_pipeline.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    sync = subcommands.add_parser("sync", help="append new source episodes")
    sync.add_argument("--config", required=True, type=Path)
    sync.add_argument("--workers", type=int, default=4)
    sync.add_argument("--limit", type=int)
    sync.add_argument("--reconvert", action="append", default=[])
    sync.add_argument("--dry-run", action="store_true")

    stats = subcommands.add_parser("stats", help="generate missing dataset statistics")
    stats.add_argument("--config", required=True, type=Path)
    stats.add_argument("--jobs", type=int)

    check = subcommands.add_parser("check", help="check converted output datasets")
    check.add_argument("--config", required=True, type=Path)
    check.add_argument("--full", action="store_true")

    train = subcommands.add_parser("train", help="prepare statistics and launch finetuning")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--jobs", type=int)
    train.add_argument("--resume-from", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "sync":
        try:
            import h5py  # noqa: F401
        except ImportError:
            print(
                "h5py is required only for sync. Run:\n"
                "  uv run --no-sync --with h5py python "
                f"{Path(__file__).as_posix()} sync --config {args.config}",
                file=sys.stderr,
            )
            return 2
        from zdata_pipeline.convert import sync_config

        return sync_config(
            config,
            workers=args.workers,
            limit=args.limit,
            reconvert=args.reconvert,
            dry_run=args.dry_run,
        )
    if args.command == "stats":
        from zdata_pipeline.check import generate_missing_stats

        return generate_missing_stats(config, jobs=args.jobs)
    if args.command == "check":
        from zdata_pipeline.check import check_outputs

        return check_outputs(config, full=args.full)
    from zdata_pipeline.check import train

    return train(config, jobs=args.jobs, resume_from=args.resume_from)


if __name__ == "__main__":
    raise SystemExit(main())
