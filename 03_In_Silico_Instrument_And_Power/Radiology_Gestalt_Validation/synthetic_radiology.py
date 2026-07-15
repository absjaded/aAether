#!/usr/bin/env python3
"""Synthetic radiology gestalt-violation cohort.

The generator models the agreed in-silico paradigm:
  * radiologist initial read
  * rare, subject-dependent gestalt violation
  * constant perceptual response independent of the violation
  * labels assigned post-hoc, while covariances are generated at read time

It returns covariance trials and public labels only. Planted axes are kept out of
the saved trial data so downstream analysis has to recover geometry blindly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import numpy as np


MODES = (
    "signal",
    "N1_null",
    "N2_response_confound",
    "N3_motor_null",
    "N4_entangled_response",
)


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 20260715
    n_subjects: int = 36
    n_trials: int = 240
    n_channels: int = 48
    pathology_rate: float = 0.42
    base_error_rate: float = 0.09
    violation_rate: float = 0.10
    gestalt_effect: float = 0.24
    response_effect: float = 0.34
    noise_scale: float = 0.080
    signal_axis_jitter: float = 0.35
    nuisance_rank: int = 6
    nuisance_effect: float = 0.10
    response_gestalt_rho: float = 0.0


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=n)
    return v / np.linalg.norm(v)


def _spd_from_low_rank(base: np.ndarray, updates: list[tuple[float, np.ndarray]]) -> np.ndarray:
    c = base.copy()
    for amp, axis in updates:
        c += amp * np.outer(axis, axis)
    c = 0.5 * (c + c.T)
    w = np.linalg.eigvalsh(c).min()
    if w <= 1e-6:
        c += np.eye(c.shape[0]) * (1e-6 - w)
    return c


def simulate(mode: str, cfg: GeneratorConfig) -> dict[str, np.ndarray | dict]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    rng = np.random.default_rng(cfg.seed + 1000 * MODES.index(mode))
    n_total = cfg.n_subjects * cfg.n_trials
    n = cfg.n_channels

    gestalt_axis = _unit(rng, n)
    response_axis = _unit(rng, n)
    response_axis -= gestalt_axis * float(response_axis @ gestalt_axis)
    response_axis /= np.linalg.norm(response_axis)
    difficulty_axis = _unit(rng, n)
    difficulty_axis -= gestalt_axis * float(difficulty_axis @ gestalt_axis)
    difficulty_axis -= response_axis * float(difficulty_axis @ response_axis)
    difficulty_axis /= np.linalg.norm(difficulty_axis)

    covs = np.empty((n_total, n, n), dtype=np.float32)
    labels: dict[str, list] = {
        "subject": [],
        "trial": [],
        "pathology": [],
        "gestalt_initial": [],
        "violation": [],
        "confidence": [],
        "divergence": [],
        "response_quality": [],
        "response_drive": [],
        "mode": [],
    }

    k = 0
    for s in range(cfg.n_subjects):
        subj_rng = np.random.default_rng(cfg.seed + 7919 * (s + 1) + 1000 * MODES.index(mode))
        q, _ = np.linalg.qr(subj_rng.normal(size=(n, n)))
        eig = np.exp(subj_rng.normal(0.0, 0.20, size=n))
        subj_base = q @ np.diag(eig) @ q.T + np.eye(n) * 0.25
        subj_error = cfg.violation_rate
        subj_gestalt_axis = gestalt_axis + cfg.signal_axis_jitter * _unit(subj_rng, n)
        subj_gestalt_axis /= np.linalg.norm(subj_gestalt_axis)
        nuisance_axes = [_unit(subj_rng, n) for _ in range(cfg.nuisance_rank)]

        for t in range(cfg.n_trials):
            pathology = int(subj_rng.random() < cfg.pathology_rate)
            difficulty = float(subj_rng.beta(2.0, 5.0))
            # Keep violation rare and subject-dependent, but do not let a scalar
            # difficulty proxy leak the label into the null covariances.
            error_prob = float(np.clip(subj_error, 0.0, 0.45))
            wrong = int(subj_rng.random() < error_prob)
            gestalt = pathology if not wrong else 1 - pathology
            violation = int(gestalt != pathology)
            confidence = float(np.clip(1.0 - difficulty + subj_rng.normal(0.0, 0.08), 0.0, 1.0))
            divergence = float(violation * (0.45 + 0.55 * confidence))
            response_drive = float(subj_rng.normal())
            if mode == "N4_entangled_response":
                rho = float(np.clip(cfg.response_gestalt_rho, -0.99, 0.99))
                p = float(np.clip(subj_error, 1e-3, 1.0 - 1e-3))
                z_violation = (violation - p) / np.sqrt(p * (1.0 - p))
                response_drive = float(
                    rho * z_violation
                    + np.sqrt(max(0.0, 1.0 - rho * rho)) * subj_rng.normal()
                )
            if response_drive < -0.43:
                response_quality = 0
            elif response_drive > 0.43:
                response_quality = 2
            else:
                response_quality = 1

            # In N4 the response axis stays geometrically orthogonal, but its
            # scalar drive is statistically entangled with the violation label.
            updates: list[tuple[float, np.ndarray]] = [
                (0.06 * difficulty, difficulty_axis),
                (cfg.noise_scale * subj_rng.normal(), _unit(subj_rng, n)),
            ]
            for axis in nuisance_axes:
                updates.append((cfg.nuisance_effect * subj_rng.normal(), axis))

            if mode == "signal":
                updates.append((cfg.gestalt_effect * divergence, subj_gestalt_axis))
            elif mode == "N2_response_confound":
                updates.append((cfg.response_effect * response_drive, response_axis))
            elif mode == "N3_motor_null":
                updates.append((cfg.gestalt_effect * divergence, subj_gestalt_axis))
                updates.append((cfg.response_effect * response_drive, response_axis))
            elif mode == "N4_entangled_response":
                updates.append((cfg.response_effect * response_drive, response_axis))
            elif mode == "N1_null":
                pass

            covs[k] = _spd_from_low_rank(subj_base, updates)
            labels["subject"].append(s)
            labels["trial"].append(t)
            labels["pathology"].append(pathology)
            labels["gestalt_initial"].append(gestalt)
            labels["violation"].append(violation)
            labels["confidence"].append(confidence)
            labels["divergence"].append(divergence)
            labels["response_quality"].append(response_quality)
            labels["response_drive"].append(response_drive)
            labels["mode"].append(mode)
            k += 1

    labels_np = {name: np.asarray(values) for name, values in labels.items()}
    public_config = dict(
        seed=cfg.seed,
        n_subjects=cfg.n_subjects,
        n_trials=cfg.n_trials,
        n_channels=cfg.n_channels,
        pathology_rate=cfg.pathology_rate,
        base_error_rate=cfg.base_error_rate,
        violation_rate=cfg.violation_rate,
        gestalt_effect=cfg.gestalt_effect,
        response_effect=cfg.response_effect,
        noise_scale=cfg.noise_scale,
        signal_axis_jitter=cfg.signal_axis_jitter,
        nuisance_rank=cfg.nuisance_rank,
        nuisance_effect=cfg.nuisance_effect,
        response_gestalt_rho=cfg.response_gestalt_rho,
        mode=mode,
    )
    return {"covariances": covs, "labels": labels_np, "config": public_config}


def save_dataset(path: Path, mode: str, cfg: GeneratorConfig) -> None:
    ds = simulate(mode, cfg)
    labels = ds["labels"]
    np.savez_compressed(
        path,
        covariances=ds["covariances"],
        **{f"label_{k}": v for k, v in labels.items()},
        config=json.dumps(ds["config"], sort_keys=True),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="03_In_Silico_Instrument_And_Power/Radiology_Gestalt_Validation/outputs")
    ap.add_argument("--mode", choices=MODES + ("all",), default="all")
    ap.add_argument("--subjects", type=int, default=GeneratorConfig.n_subjects)
    ap.add_argument("--trials", type=int, default=GeneratorConfig.n_trials)
    ap.add_argument("--channels", type=int, default=GeneratorConfig.n_channels)
    ap.add_argument("--gestalt-effect", type=float, default=GeneratorConfig.gestalt_effect)
    ap.add_argument("--response-effect", type=float, default=GeneratorConfig.response_effect)
    ap.add_argument("--violation-rate", type=float, default=GeneratorConfig.violation_rate)
    ap.add_argument("--signal-axis-jitter", type=float, default=GeneratorConfig.signal_axis_jitter)
    ap.add_argument("--noise-scale", type=float, default=GeneratorConfig.noise_scale)
    ap.add_argument("--nuisance-rank", type=int, default=GeneratorConfig.nuisance_rank)
    ap.add_argument("--nuisance-effect", type=float, default=GeneratorConfig.nuisance_effect)
    ap.add_argument("--response-gestalt-rho", type=float, default=GeneratorConfig.response_gestalt_rho)
    args = ap.parse_args()

    cfg = GeneratorConfig(
        n_subjects=args.subjects,
        n_trials=args.trials,
        n_channels=args.channels,
        gestalt_effect=args.gestalt_effect,
        response_effect=args.response_effect,
        violation_rate=args.violation_rate,
        signal_axis_jitter=args.signal_axis_jitter,
        noise_scale=args.noise_scale,
        nuisance_rank=args.nuisance_rank,
        nuisance_effect=args.nuisance_effect,
        response_gestalt_rho=args.response_gestalt_rho,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = MODES if args.mode == "all" else (args.mode,)
    for mode in modes:
        save_dataset(out / f"{mode}.npz", mode, cfg)


if __name__ == "__main__":
    main()
