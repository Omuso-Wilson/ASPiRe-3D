"""
ASPiRe-3D : main/run_aspire3d.py
===============================================================================
THE single top-level entry point for ASPiRe-3D.

Everything is driven by a JSON configuration file: core/fluid/ASP properties,
injection schedule, numerics, physics toggles, and calibration/sensitivity/
uncertainty/output controls. No experimental values are hardcoded.

USAGE
-----
  python -m main.run_aspire3d --config experiments/configs/baseline_case.json
  python -m main.run_aspire3d --config experiments/configs/validation_case.json
  python -m main.run_aspire3d --config <your_case>.json --quiet

  # list the bundled example configs
  python -m main.run_aspire3d --list

The entry point only: parses CLI args, loads+validates the config, hands control
to the Workflow controller, and reports where outputs were written. All physics,
calibration, and plotting live in their respective packages.
===============================================================================
"""

import os
import sys
import argparse

# Ensure the project root is importable when run as a script or module.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from main.config_loader import load_config, ConfigError
from main.workflow import Workflow


EXAMPLE_DIR = os.path.join(_ROOT, "experiments", "configs")


def _list_configs():
    print("Bundled example configurations (experiments/configs/):")
    if os.path.isdir(EXAMPLE_DIR):
        for f in sorted(os.listdir(EXAMPLE_DIR)):
            if f.endswith(".json"):
                print(f"  - {f}")
    print("\nRun one with:\n  python -m main.run_aspire3d --config "
          "experiments/configs/baseline_case.json")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="run_aspire3d",
        description="ASPiRe-3D — config-driven ASP formation-damage simulator.")
    parser.add_argument("--config", "-c", help="path to a JSON config file")
    parser.add_argument("--list", "-l", action="store_true",
                        help="list bundled example configs and exit")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="suppress console step output (still logs to file)")
    args = parser.parse_args(argv)

    if args.list or not args.config:
        _list_configs()
        return 0 if args.list else 1

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"[config error] {e}", file=sys.stderr)
        return 2

    print("=" * 60)
    print(f"ASPiRe-3D  |  {cfg.raw.get('name', os.path.basename(args.config))}")
    print("=" * 60)
    workflow = Workflow(cfg, verbose=not args.quiet)
    try:
        results = workflow.run()
    except Exception as e:
        print(f"[workflow error] {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    print("-" * 60)
    print(f"outputs written to: {workflow.outdir}/")
    if results.get("report"):
        print(f"  report : {os.path.basename(results['report'])}")
    if results.get("figures"):
        print(f"  figures: {len(results['figures'])}")
    print(f"  log    : {os.path.basename(results['logfile'])}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
