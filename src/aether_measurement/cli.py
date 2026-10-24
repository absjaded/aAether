"""Command-line entry points; outputs stay outside the packaged evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from zipfile import BadZipFile


def _write(report, output):
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main(argv=None):
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Preserve the ledger/baseline argument contracts and their own help.
    if args_list and args_list[0] == "ledger":
        from .ledger import main as ledger_main
        return ledger_main(args_list[1:])
    if args_list and args_list[0] == "baseline":
        from .baseline import main as baseline_main
        return baseline_main(args_list[1:])
    parser = argparse.ArgumentParser(description="Aether covariance measurement and control experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run four small synthetic signal/nuisance constructions")
    demo.add_argument("--subjects", type=int, default=6)
    demo.add_argument("--trials", type=int, default=72, help="trials per subject")
    demo.add_argument("--channels", type=int, default=8)
    demo.add_argument("--seed", type=int, default=20260715)
    demo.add_argument("--out", type=Path)
    demo.add_argument("--save-trials", type=Path, help="directory for generated NPZ inputs")
    measurement = sub.add_parser("measure", help="analyze covariances, labels, subjects from an NPZ file")
    measurement.add_argument("--input", type=Path, required=True)
    measurement.add_argument("--components", type=int, default=12)
    measurement.add_argument("--splits", type=int, default=6)
    measurement.add_argument("--out", type=Path)
    measurement.add_argument("--include-scores", action="store_true")
    records = sub.add_parser("records", help="check and reaggregate packaged historical numerical records")
    records.add_argument("--out", type=Path)
    sub.add_parser("ledger", help="initialize or validate behavioral CSV ledgers")
    sub.add_parser("baseline", help="estimate cheap-observable behavioral baselines")
    args = parser.parse_args(args_list)
    try:
        if args.command == "records":
            from .records import replay_records
            report = replay_records()
        else:
            import numpy as np
            from .instrument import grouped_readout, measure
            if args.command == "measure":
                loaded = np.load(args.input, allow_pickle=False)
                if not isinstance(loaded, np.lib.npyio.NpzFile):
                    raise ValueError("--input must be an NPZ archive with covariances, labels and subjects")
                with loaded as data:
                    report = measure(data["covariances"], data["labels"], data["subjects"],
                                     components=args.components, splits=args.splits)
                if not args.include_scores:
                    report.pop("scores")
                    report.pop("fold_ids")
            else:
                from .simulation import MODES, SimulationConfig, simulate
                config = SimulationConfig(seed=args.seed, subjects=args.subjects, trials=args.trials, channels=args.channels)
                report = {"kind": "new synthetic demonstration", "config": {}, "scenarios": []}
                for mode in MODES:
                    data = simulate(mode, config)
                    neural = measure(data["covariances"], data["labels"], data["subjects"])
                    proxy = grouped_readout(data["response_proxy"][:, None], data["labels"], data["subjects"])
                    report["scenarios"].append({"mode": mode, "covariance_auc": neural["auc"],
                        "response_proxy_auc": proxy["auc"], "trials": neural["trials"],
                        "positives": neural["positives"], "subjects": neural["subjects"]})
                    report["config"] = {key: value for key, value in data["config"].items() if key != "mode"}
                    if args.save_trials:
                        args.save_trials.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(args.save_trials / f"{mode}.npz", **{
                            key: data[key] for key in ("covariances", "labels", "subjects", "response_proxy")})
                report["protocol"] = neural["protocol"]
                report["calibration"] = neural["calibration"]
        _write(report, args.out)
    except (ValueError, OSError, KeyError, BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
