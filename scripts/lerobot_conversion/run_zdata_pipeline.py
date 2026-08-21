#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the config-driven zdata_hdf5 to GR00T pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from zdata_pipeline.common import frozen_corpus_manifest_path
from zdata_pipeline.config import PipelineConfig, load_config


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)


def _add_train_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--smoke-max-steps", type=int)
    parser.add_argument("--smoke-batch", type=int)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_config(parser)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reconvert", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke-max-steps", type=int)
    parser.add_argument("--smoke-batch", type=int, default=1)
    subcommands = parser.add_subparsers(dest="command")

    sync = subcommands.add_parser(
        "sync", help="append new source episodes", argument_default=argparse.SUPPRESS
    )
    _add_config(sync)
    sync.add_argument("--workers", type=int)
    sync.add_argument("--limit", type=int)
    sync.add_argument("--reconvert", action="append")
    sync.add_argument("--dry-run", action="store_true")

    stats = subcommands.add_parser(
        "stats", help="generate missing dataset statistics", argument_default=argparse.SUPPRESS
    )
    _add_config(stats)
    stats.add_argument("--jobs", type=int)

    check = subcommands.add_parser(
        "check", help="check converted output datasets", argument_default=argparse.SUPPRESS
    )
    _add_config(check)
    check.add_argument("--full", action="store_true")

    freeze = subcommands.add_parser(
        "freeze", help="content-bind a corpus against writes", argument_default=argparse.SUPPRESS
    )
    _add_config(freeze)

    train = subcommands.add_parser(
        "train",
        help="prepare statistics and launch finetuning",
        argument_default=argparse.SUPPRESS,
    )
    _add_config(train)
    _add_train_options(train)
    return parser


def _require_h5py(config_path: Path) -> int | None:
    try:
        import h5py  # noqa: F401
    except ImportError:
        print(
            "h5py is required for sync. Run:\n"
            "  uv run --no-sync --with h5py python "
            f"{Path(__file__).as_posix()} --config {config_path}",
            file=sys.stderr,
        )
        return 2
    return None


def _unfreeze_for_experiment(config: PipelineConfig) -> None:
    marker = frozen_corpus_manifest_path(config.output.root)
    if marker.exists():
        marker.unlink()
        print(f"removed freeze marker for mutable experimental pipeline: {marker}")


def _run_default(args: argparse.Namespace, config_path: Path) -> int:
    missing_dependency = _require_h5py(config_path)
    if missing_dependency is not None:
        return missing_dependency
    from zdata_pipeline.check import check_outputs, freeze_corpus, generate_missing_stats, train
    from zdata_pipeline.convert import sync_config

    config = load_config(config_path)
    marker = frozen_corpus_manifest_path(config.output.root)
    if args.freeze and marker.exists():
        print(f"using existing frozen corpus and skipping sync: {marker}")
    else:
        _unfreeze_for_experiment(config)
        result = sync_config(
            config,
            workers=args.workers,
            limit=args.limit,
            reconvert=args.reconvert,
            dry_run=args.dry_run,
        )
        if result:
            return result
    if args.dry_run:
        return 0
    for stage in (
        lambda: generate_missing_stats(config, jobs=args.jobs),
        lambda: check_outputs(config, full=args.full),
    ):
        result = stage()
        if result:
            return result
    if args.freeze:
        result = freeze_corpus(config)
        if result:
            return result
    return train(
        config,
        jobs=args.jobs,
        resume_from=args.resume_from,
        smoke_max_steps=args.smoke_max_steps,
        smoke_batch=args.smoke_batch,
        require_frozen=args.freeze,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config_path = args.config
    if config_path is None:
        parser.error("--config is required")
    if args.command is None:
        return _run_default(args, config_path)

    config = load_config(config_path)
    if args.command == "sync":
        missing_dependency = _require_h5py(config_path)
        if missing_dependency is not None:
            return missing_dependency
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
    if args.command == "freeze":
        from zdata_pipeline.check import freeze_corpus

        return freeze_corpus(config)

    from zdata_pipeline.check import freeze_corpus, train

    if not args.freeze:
        _unfreeze_for_experiment(config)
    elif result := freeze_corpus(config):
        return result
    return train(
        config,
        jobs=args.jobs,
        resume_from=args.resume_from,
        smoke_max_steps=args.smoke_max_steps,
        smoke_batch=args.smoke_batch,
        require_frozen=args.freeze,
    )


if __name__ == "__main__":
    raise SystemExit(main())
