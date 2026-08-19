# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small pipeline contracts shared by HDF5 and output-only commands."""

from __future__ import annotations

from pathlib import Path


STATS_FILES = ("stats.json", "relative_stats.json")


def transaction_journals(output_root: Path) -> list[Path]:
    if not output_root.is_dir():
        return []
    return sorted(output_root.glob("*/.sync_transaction.json"))


def assert_no_incomplete_transactions(output_root: Path) -> None:
    journals = transaction_journals(output_root)
    if journals:
        joined = ", ".join(str(path) for path in journals)
        raise RuntimeError(f"incomplete sync transaction found: {joined}; run sync to recover it")
