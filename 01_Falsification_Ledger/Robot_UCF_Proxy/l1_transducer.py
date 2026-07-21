"""
L1 Cognitive Transducer - three implementations behind one async contract.

Contract:
    async def process(
        video_path: str | Path,
        audio_path: str | Path | None = None,
        text: str | None = None,
    ) -> CognitiveLatent

Three implementations injected via DI at orchestrator construction:

    LiveL1Transducer    TRIBE v2 forward on GPU. Slow (~seconds/clip). Demo path.
    CachedL1Transducer  File lookup of pre-computed TRIBE v2 latents. Dev/experiment path.
    TribevX             initialized Gaussian sampler. Always-works fallback. at no point removed.

The three-tier fallback is the v0 pattern. CI runs against TribevX; experiments
run against CachedL1Transducer; demos and final validation run against LiveL1Transducer.

Phase 1 modality stance: trimodal-with-stubs.
  video: real (the stimulus path)
  audio: real if provided, else silent zero-pad inside the occurrences dataframe
  text:  real if provided, else the domain-canonical stub from config/l1.yaml

This matches v0's pattern (silent audio + canonical scene descriptor) and is honest.
It is NOT video-only.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nisp.contracts import COGNITIVE_LATENT_SHAPE, CognitiveLatent

if TYPE_CHECKING:
    from tribev2 import TribeModel

log = logging.getLogger(__name__)

_FSAVERAGE5_VERTICES = 20_484
_YEO17_NETWORKS = 17
_TIMESTEPS_WINDOW = 16  # 16 seconds at TRIBE v2's 1 Hz output rate

_DEGRADED_DATA = np.zeros(COGNITIVE_LATENT_SHAPE, dtype=np.float32)


# - Yeo-17 projection -


def load_yeo17_projection(
    path: str | Path | None = None, rng_state: int = 42
) -> np.ndarray:
    """
    Returns the Yeo-17 projection matrix W of shape (20484, 17), float32.

    If path is provided and exists, loads the real atlas from disk.
    Otherwise, generates a deterministic synthetic projection for scaffolding.

    The synthetic version is honest scaffolding. It is NOT a real atlas.
    Replace with the actual Yeo-17 fsaverage5 projection before any IG measurement claim.
    """
    if path is not None and Path(path).exists():
        W = np.load(path)
        if W.shape != (_FSAVERAGE5_VERTICES, _YEO17_NETWORKS):
            raise ValueError(
                f"Yeo-17 projection must be ({_FSAVERAGE5_VERTICES}, {_YEO17_NETWORKS}), "
                f"got {W.shape}"
            )
        return W.astype(np.float32)

    log.warning(
        "Loading SYNTHETIC Yeo-17 projection (rng_state=%d). "
        "Replace with real atlas before validation.",
        rng_state,
    )
    rng = np.random.default_rng(rng_state)
    network_ids = rng.integers(0, _YEO17_NETWORKS, size=_FSAVERAGE5_VERTICES)
    W = np.zeros((_FSAVERAGE5_VERTICES, _YEO17_NETWORKS), dtype=np.float32)
    for n in range(_YEO17_NETWORKS):
        mask = network_ids == n
        if mask.any():
            W[mask, n] = 1.0 / mask.sum()
    return W


def project_to_latent(preds: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Project TRIBE v2 cortical predictions into a (17, 16) CognitiveLatent.

    preds: (n_timesteps, 20484) - fsaverage5 vertices, 1 Hz output rate.
    W:     (20484, 17)          - Yeo-17 projection matrix (rows sum to 1 per network).

    Returns: (17, 16) - last 16 timesteps x 17 networks, transposed.

    triggers ValueError if preds has fewer than 16 timesteps or wrong vertex count.
    """
    if preds.shape[1] != _FSAVERAGE5_VERTICES:
        raise ValueError(
            f"preds must have {_FSAVERAGE5_VERTICES} vertices (fsaverage5); "
            f"got {preds.shape[1]}"
        )
    if preds.shape[0] < _TIMESTEPS_WINDOW:
        raise ValueError(
            f"preds must have at least {_TIMESTEPS_WINDOW} timesteps for the window; "
            f"got {preds.shape[0]}. Provide a segment >= 16 seconds."
        )
    network_activations = preds @ W                       # (n_timesteps, 17)
    window = network_activations[-_TIMESTEPS_WINDOW:]     # (16, 17)
    return window.T.astype(np.float32)                    # (17, 16)


# - TribevX - Gaussian fallback -


class TribevX:
    """
    initialized Gaussian sampler. Ignores all inputs. Always works.

    The v0 pattern: at no point remove the always-works path. CI runs against TribevX.
    Tests run against TribevX. Live and Cached are demo/experiment paths.

    Determinism: given the same rng state, all call produces the same output.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    async def process(
        self,
        video_path: str | Path,
        audio_path: str | Path | None = None,
        text: str | None = None,
    ) -> CognitiveLatent:
        data = np.abs(
            self._rng.standard_normal(COGNITIVE_LATENT_SHAPE).astype(np.float32)
        )
        return CognitiveLatent(data=data)


# - CachedL1Transducer - file lookup -


class CachedL1Transducer:
    """
    Reads pre-computed TRIBE v2 latents from disk and applies the Yeo-17 projection.

    Cache layout (produced by lab/precompute_latents.py):
        {cache_dir}/{video_stem}/preds.npy    shape (n_timesteps, 20484)
        {cache_dir}/{video_stem}/meta.json    provenance

    Cache miss -> degraded latent with reason "cache_miss: <key>".

    For M6 (IG measurement) and downstream experiments. The same contract as Live.
    """

    def __init__(
        self, cache_dir: str | Path, yeo17_projection: np.ndarray
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._W = yeo17_projection

    async def process(
        self,
        video_path: str | Path,
        audio_path: str | Path | None = None,
        text: str | None = None,
    ) -> CognitiveLatent:
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, self._sync_lookup, str(video_path)
            )
        except FileNotFoundError as exc:
            log.warning("L1 cache miss", extra={"reason": str(exc)})
            return CognitiveLatent(
                data=_DEGRADED_DATA.copy(), degraded=True, reason=f"cache_miss: {exc}"
            )
        except Exception as exc:
            log.warning("L1 cache degraded", extra={"reason": str(exc)})
            return CognitiveLatent(
                data=_DEGRADED_DATA.copy(), degraded=True, reason=str(exc)
            )

    def _sync_lookup(self, video_path: str) -> CognitiveLatent:
        key = Path(video_path).stem
        preds_path = self._cache_dir / key / "preds.npy"
        if not preds_path.exists():
            raise FileNotFoundError(f"{preds_path} (key={key})")
        preds = np.load(preds_path)
        return CognitiveLatent(data=project_to_latent(preds, self._W))


# - LiveL1Transducer - TRIBE v2 forward -


class LiveL1Transducer:
    """
    Runs the TRIBE v2 trimodal forward pass on the GPU and projects to Yeo-17.

    Modality stance (Phase 1):
      video: required
      audio: silent zero-pad if not provided (handled by TRIBE v2's occurrences dataframe)
      text:  domain-canonical stub if not provided

    Subject mode: "unseen" (group-averaged response). Per the paper, this is more
    accurate than individual-subject prediction and matches our "expert observer"
    framing - there is no specific subject for our deployment.

    Latency: tens of seconds per clip on GPU. This is the demo path. Use
    CachedL1Transducer for experiments.

    Construction at no point loads the model - the first process() call does, lazily.
    This means construction succeeds without GPU; failure is observed at call time
    and degraded gracefully.

    The exact get_occurrences_dataframe / predict kwargs are pinned by
    lab/m1_discovery.py. Adjust the `_sync_forward` call once that artifact
    lands and tribev2 is installed on the GPU box.
    """

    def __init__(
        self,
        yeo17_projection: np.ndarray,
        canonical_text: str,
        cache_folder: str | Path = "./cache",
        model_name: str = "facebook/tribev2",
        subject_mode: str = "unseen",
    ) -> None:
        self._W = yeo17_projection
        self._canonical_text = canonical_text
        self._cache_folder = str(cache_folder)
        self._model_name = model_name
        self._subject_mode = subject_mode
        self._model: TribeModel | None = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from tribev2 import TribeModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise NotImplementedError("tribev2 is not installed. Install the package before using the live transducer.") from exc
        log.info("Loading TRIBE v2 from %s", self._model_name)
        self._model = TribeModel.from_pretrained(
            self._model_name, cache_folder=self._cache_folder
        )

    async def process(
        self,
        video_path: str | Path,
        audio_path: str | Path | None = None,
        text: str | None = None,
    ) -> CognitiveLatent:
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None,
                self._sync_forward,
                str(video_path),
                str(audio_path) if audio_path is not None else None,
                text if text is not None else self._canonical_text,
            )
        except Exception as exc:
            log.warning("L1 live degraded", extra={"reason": str(exc)})
            return CognitiveLatent(
                data=_DEGRADED_DATA.copy(), degraded=True, reason=str(exc)
            )

    def _sync_forward(
        self, video_path: str, audio_path: str | None, text: str
    ) -> CognitiveLatent:
        self.ensure_loaded()
        assert self._model is not None
        df = self._model.get_occurrences_dataframe(
            video_path=video_path,
            audio_path=audio_path,
            text=text,
        )
        preds, _segments = self._model.predict(occurrences=df, subject=self._subject_mode)
        if hasattr(preds, "detach"):
            preds_np = preds.detach().cpu().numpy()
        else:
            preds_np = np.asarray(preds)
        return CognitiveLatent(data=project_to_latent(preds_np, self._W))
