"""
Grid-search kernel enable flags in resnet18_engine.cpp.

Each configuration patches the constexpr bool enable_* flags in both the
conv_bn_relu and conv_bn dispatch functions, rebuilds the binary, runs the
benchmark, and records timing. A summary table is printed at the end.

Usage:
    python scripts/tile_search.py [--threads N] [--runs N] [--warmup N]
                                  [--engine-src PATH] [--build-dir PATH]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration space
# ---------------------------------------------------------------------------

# Each entry defines one configuration to benchmark.
# "flags" is a dict of flag name → bool.  Flags not listed here are left at
# their on-disk defaults so only the ones we care about are changed.
CONFIGS: list[dict[str, Any]] = [
    {
        "name": "baseline",
        "desc": "OC2+OH2 spatial2 for L1-3, OC4 L4-specific, Q8 L4-s1",
        "flags": {
            "enable_pattern_ocblock2_spatial2_kernel": True,
            "enable_pattern_ocblock4_layer4_kernel": True,
            "enable_pattern_ocblock4_general_s1_kernel": False,
            "enable_q8_layer4_stride1_kernel": True,
            "enable_q8_layer3_s2_kernel": False,
            "enable_conv1_ocblock4_kernel": True,
        },
    },
    {
        "name": "no_spatial2",
        "desc": "OC2+OH1 only for L1-3 (no row-pair kernel)",
        "flags": {
            "enable_pattern_ocblock2_spatial2_kernel": False,
            "enable_pattern_ocblock4_layer4_kernel": True,
            "enable_pattern_ocblock4_general_s1_kernel": False,
            "enable_q8_layer4_stride1_kernel": True,
            "enable_q8_layer3_s2_kernel": False,
            "enable_conv1_ocblock4_kernel": True,
        },
    },
    {
        "name": "oc4_general_plus_spatial2",
        "desc": "OC4+OH2 runtime-cond for L1-3 BEFORE spatial2 fallback",
        "flags": {
            "enable_pattern_ocblock2_spatial2_kernel": True,
            "enable_pattern_ocblock4_layer4_kernel": True,
            "enable_pattern_ocblock4_general_s1_kernel": True,
            "enable_q8_layer4_stride1_kernel": True,
            "enable_q8_layer3_s2_kernel": False,
            "enable_conv1_ocblock4_kernel": True,
        },
    },
    {
        "name": "oc4_general_no_spatial2",
        "desc": "OC4+OH2 runtime-cond only (no spatial2 fallback) for L1-3",
        "flags": {
            "enable_pattern_ocblock2_spatial2_kernel": False,
            "enable_pattern_ocblock4_layer4_kernel": True,
            "enable_pattern_ocblock4_general_s1_kernel": True,
            "enable_q8_layer4_stride1_kernel": True,
            "enable_q8_layer3_s2_kernel": False,
            "enable_conv1_ocblock4_kernel": True,
        },
    },
    {
        "name": "no_q8_layer4",
        "desc": "Baseline but disable Q8 for L4 stride-1 (FP32 OC4)",
        "flags": {
            "enable_pattern_ocblock2_spatial2_kernel": True,
            "enable_pattern_ocblock4_layer4_kernel": True,
            "enable_pattern_ocblock4_general_s1_kernel": False,
            "enable_q8_layer4_stride1_kernel": False,
            "enable_q8_layer3_s2_kernel": False,
            "enable_conv1_ocblock4_kernel": True,
        },
    },
    {
        "name": "q8_layer3_s2_enabled",
        "desc": "Baseline + enable Q8 for L3 stride-2 (currently disabled)",
        "flags": {
            "enable_pattern_ocblock2_spatial2_kernel": True,
            "enable_pattern_ocblock4_layer4_kernel": True,
            "enable_pattern_ocblock4_general_s1_kernel": False,
            "enable_q8_layer4_stride1_kernel": True,
            "enable_q8_layer3_s2_kernel": True,
            "enable_conv1_ocblock4_kernel": True,
        },
    },
    {
        "name": "no_oc4_layer4",
        "desc": "Disable specialised OC4 L4 kernel (falls through to spatial2/OC2)",
        "flags": {
            "enable_pattern_ocblock2_spatial2_kernel": True,
            "enable_pattern_ocblock4_layer4_kernel": False,
            "enable_pattern_ocblock4_general_s1_kernel": False,
            "enable_q8_layer4_stride1_kernel": True,
            "enable_q8_layer3_s2_kernel": False,
            "enable_conv1_ocblock4_kernel": True,
        },
    },
    {
        "name": "minimal_oc2_oh1",
        "desc": "Only basic OC2 pattern kernel, no spatial2, no OC4, no Q8",
        "flags": {
            "enable_pattern_ocblock2_spatial2_kernel": False,
            "enable_pattern_ocblock4_layer4_kernel": False,
            "enable_pattern_ocblock4_general_s1_kernel": False,
            "enable_q8_layer4_stride1_kernel": False,
            "enable_q8_layer3_s2_kernel": False,
            "enable_conv1_ocblock4_kernel": False,
        },
    },
]

# ---------------------------------------------------------------------------
# Source patching helpers
# ---------------------------------------------------------------------------

def _patch_flag(source: str, flag: str, value: bool) -> str:
    """Replace all occurrences of `constexpr bool <flag> = …` in source."""
    pat = re.compile(
        r"(constexpr\s+bool\s+" + re.escape(flag) + r"\s*=\s*)(true|false)(;)"
    )
    replacement = r"\g<1>" + ("true" if value else "false") + r"\g<3>"
    patched, n = pat.subn(replacement, source)
    if n == 0:
        raise RuntimeError(f"Flag '{flag}' not found in source")
    return patched


def apply_flags(src_path: Path, flags: dict[str, bool]) -> None:
    original = src_path.read_text(encoding="utf-8")
    patched = original
    for flag, value in flags.items():
        patched = _patch_flag(patched, flag, value)
    src_path.write_text(patched, encoding="utf-8")


def restore_from_backup(src_path: Path, backup_path: Path) -> None:
    shutil.copy2(backup_path, src_path)


# ---------------------------------------------------------------------------
# Build and benchmark helpers
# ---------------------------------------------------------------------------

def rebuild(build_dir: Path, jobs: int) -> tuple[bool, str]:
    """Rebuild basic_engine. Returns (success, stderr)."""
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "basic_engine", f"-j{jobs}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


LATENCY_BLOCK_RE = re.compile(
    r"C\+\+\s+engine\s+latency\s+"
    r"runs:\s+\d+\s+"
    r"min_ms:\s+([0-9.]+)\s+"
    r"median_ms:\s+([0-9.]+)\s+"
    r"mean_ms:\s+([0-9.]+)\s+"
    r"p90_ms:\s+([0-9.]+)",
    re.MULTILINE,
)


def run_benchmark(
    engine_path: Path,
    model_bin: Path,
    input_bin: Path,
    threads: int,
    runs: int,
) -> dict[str, float] | None:
    env_overrides = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_NESTED": "FALSE",
    }
    import os
    env = {**os.environ, **env_overrides}

    result = subprocess.run(
        [str(engine_path), str(model_bin), str(input_bin)],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        return None

    m = LATENCY_BLOCK_RE.search(result.stdout)
    if m is None:
        return None

    return {
        "min_ms": float(m.group(1)),
        "median_ms": float(m.group(2)),
        "mean_ms": float(m.group(3)),
        "p90_ms": float(m.group(4)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    desc: str
    flags: dict[str, bool]
    stats: dict[str, float] | None = None
    build_ok: bool = True
    build_err: str = ""
    elapsed_s: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-src",
        type=Path,
        default=Path("src/basic_engine/resnet18_engine.cpp"),
        help="Source file containing the constexpr bool enable_* flags",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("build/basic_engine"),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="OMP_NUM_THREADS for benchmarking (default: 1)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel compile jobs (default: nproc)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of inference runs per config (default: 100)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Subset of config names to run (default: all)",
    )
    args = parser.parse_args()

    import os
    jobs = args.jobs or os.cpu_count() or 4

    model_bin = args.artifact_dir / "model_ternary_i8_weights_v1.bin"
    input_bin = args.artifact_dir / "golden_input_fp32.bin"

    for p in [model_bin, input_bin, args.engine_src, args.build_dir]:
        if not p.exists():
            print(f"ERROR: required path not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Make a backup of the source file to restore afterwards.
    backup = args.engine_src.with_suffix(".cpp.tile_search_bak")
    shutil.copy2(args.engine_src, backup)
    print(f"Backed up {args.engine_src} → {backup}\n")

    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = [c for c in CONFIGS if c["name"] in args.configs]
        if not configs_to_run:
            print("ERROR: no matching configs found", file=sys.stderr)
            sys.exit(1)

    results: list[Result] = []

    try:
        for cfg in configs_to_run:
            name = cfg["name"]
            print(f"{'='*60}")
            print(f"Config: {name}")
            print(f"  {cfg['desc']}")
            for k, v in cfg["flags"].items():
                print(f"  {k} = {v}")

            r = Result(name=name, desc=cfg["desc"], flags=cfg["flags"])

            # Patch source
            restore_from_backup(args.engine_src, backup)
            apply_flags(args.engine_src, cfg["flags"])

            # Build
            t0 = time.perf_counter()
            ok, err = rebuild(args.build_dir, jobs)
            r.build_ok = ok
            r.build_err = err
            if not ok:
                print(f"  BUILD FAILED:\n{err}")
                results.append(r)
                continue

            # Benchmark
            stats = run_benchmark(
                engine_path=args.engine,
                model_bin=model_bin,
                input_bin=input_bin,
                threads=args.threads,
                runs=args.runs,
            )
            r.elapsed_s = time.perf_counter() - t0
            r.stats = stats

            if stats:
                print(
                    f"  min={stats['min_ms']:.2f}ms  "
                    f"median={stats['median_ms']:.2f}ms  "
                    f"mean={stats['mean_ms']:.2f}ms  "
                    f"p90={stats['p90_ms']:.2f}ms  "
                    f"({r.elapsed_s:.0f}s total)"
                )
            else:
                print("  BENCHMARK FAILED (engine crashed or parse error)")
            print()
            results.append(r)

    finally:
        # Always restore original source and rebuild so repo stays clean.
        print("Restoring original source file and rebuilding...")
        restore_from_backup(args.engine_src, backup)
        rebuild(args.build_dir, jobs)
        backup.unlink(missing_ok=True)
        print("Done.\n")

    # Summary table
    best_median = min(
        (r.stats["median_ms"] for r in results if r.stats), default=None
    )

    header = f"{'Config':<35} {'min':>8} {'median':>8} {'p90':>8}  {'vs best':>8}  Description"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for r in results:
        if not r.build_ok:
            row = f"{'[BUILD FAIL] ' + r.name:<35} {'—':>8} {'—':>8} {'—':>8}  {'—':>8}  {r.desc}"
        elif r.stats is None:
            row = f"{'[BENCH FAIL] ' + r.name:<35} {'—':>8} {'—':>8} {'—':>8}  {'—':>8}  {r.desc}"
        else:
            s = r.stats
            vs = ""
            if best_median is not None:
                pct = (s["median_ms"] / best_median - 1.0) * 100.0
                vs = f"+{pct:.1f}%" if pct > 0.1 else "BEST"
            row = (
                f"{r.name:<35} "
                f"{s['min_ms']:>8.2f} "
                f"{s['median_ms']:>8.2f} "
                f"{s['p90_ms']:>8.2f}  "
                f"{vs:>8}  "
                f"{r.desc}"
            )
        print(row)
    print("=" * len(header))


if __name__ == "__main__":
    main()
