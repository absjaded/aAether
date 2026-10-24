"""Small covariance constructions adapted from synthetic_radiology.py.

This portable generator retains the historical signal/null/response mechanisms
with explicit small defaults. It is a new demonstration, not a replay of the
historical cohorts. Its labels and response proxy are constructed variables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

MODES = ("signal", "independent_null", "response_only", "entangled_response")


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 20260715
    subjects: int = 6
    trials: int = 72
    channels: int = 8
    violation_rate: float = 0.10
    signal_effect: float = 0.24
    response_effect: float = 0.34
    response_label_rho: float = 0.30
    noise_scale: float = 0.08
    nuisance_effect: float = 0.10
    axis_jitter: float = 0.35


def _unit(rng, channels):
    vector = rng.normal(size=channels)
    return vector / np.linalg.norm(vector)


def simulate(mode: str, config: SimulationConfig = SimulationConfig()) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if any(not isinstance(value, int) for value in (config.seed, config.subjects, config.trials, config.channels)):
        raise ValueError("seed, subjects, trials and channels must be integers")
    if config.seed < 0 or config.subjects < 2 or config.trials < 2 or config.channels < 3:
        raise ValueError("require nonnegative seed, at least two subjects/trials and three channels")
    numeric = [config.violation_rate, config.signal_effect, config.response_effect,
               config.response_label_rho, config.noise_scale, config.nuisance_effect, config.axis_jitter]
    if not np.isfinite(numeric).all() or not 0 < config.violation_rate < 1 or abs(config.response_label_rho) >= 1:
        raise ValueError("finite parameters, 0 < violation_rate < 1 and abs(rho) < 1 required")
    if min(config.signal_effect, config.response_effect, config.noise_scale, config.nuisance_effect, config.axis_jitter) < 0:
        raise ValueError("effect, noise and jitter magnitudes must be nonnegative")

    offset = 1000 * MODES.index(mode)
    rng = np.random.default_rng(config.seed + offset)
    signal_axis = _unit(rng, config.channels)
    response_axis = _unit(rng, config.channels)
    response_axis -= signal_axis * (response_axis @ signal_axis)
    response_axis /= np.linalg.norm(response_axis)
    difficulty_axis = _unit(rng, config.channels)
    difficulty_axis -= signal_axis * (difficulty_axis @ signal_axis)
    difficulty_axis -= response_axis * (difficulty_axis @ response_axis)
    difficulty_axis /= np.linalg.norm(difficulty_axis)
    covariances, labels, subjects, response_proxies = [], [], [], []
    for subject in range(config.subjects):
        srng = np.random.default_rng(config.seed + 7919 * (subject + 1) + offset)
        q, _ = np.linalg.qr(srng.normal(size=(config.channels, config.channels)))
        base = (q * np.exp(srng.normal(0, 0.20, config.channels))) @ q.T + 0.25 * np.eye(config.channels)
        local_axis = signal_axis + config.axis_jitter * _unit(srng, config.channels)
        local_axis /= np.linalg.norm(local_axis)
        nuisance_axes = [_unit(srng, config.channels) for _ in range(3)]
        for _ in range(config.trials):
            label = int(srng.random() < config.violation_rate)
            difficulty = srng.beta(2, 5)
            confidence = np.clip(1 - difficulty + srng.normal(0, 0.08), 0, 1)
            response = srng.normal()
            if mode == "entangled_response":
                standardized_label = (label - config.violation_rate) / np.sqrt(config.violation_rate * (1 - config.violation_rate))
                rho = config.response_label_rho
                response = rho * standardized_label + np.sqrt(1 - rho * rho) * srng.normal()
            updates = [(0.06 * difficulty, difficulty_axis), (config.noise_scale * srng.normal(), _unit(srng, config.channels))]
            updates.extend((config.nuisance_effect * srng.normal(), axis) for axis in nuisance_axes)
            if mode == "signal":
                updates.append((config.signal_effect * label * (0.45 + 0.55 * confidence), local_axis))
            if mode in ("response_only", "entangled_response"):
                updates.append((config.response_effect * response, response_axis))
            covariance = base.copy()
            for amplitude, axis in updates:
                covariance += amplitude * np.outer(axis, axis)
            covariance = 0.5 * covariance + 0.5 * covariance.T
            smallest = np.linalg.eigvalsh(covariance)[0]
            if smallest <= 1e-6:
                covariance += np.eye(config.channels) * (1e-6 - smallest)
            covariances.append(covariance)
            labels.append(label)
            subjects.append(subject)
            response_proxies.append(response)
    return {
        "covariances": np.asarray(covariances), "labels": np.asarray(labels),
        "subjects": np.asarray(subjects), "response_proxy": np.asarray(response_proxies),
        "config": {**asdict(config), "mode": mode},
    }
