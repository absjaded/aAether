from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from nisp.certificate import generate_certificate
from nisp.contracts import (
    Action,
    CognitiveLatent,
    EDSLExpression,
    SafetyVerdict,
)
from nisp.l3_proof import L3proofSolver
from nisp.l5_arbiter import L5Arbiter
from nisp.safety_instruments import ECEMonitor, IGMonitor

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleResult:
    cycle_id: str
    verdict: SafetyVerdict
    certificate_root: str
    latency_ms: float
    any_degraded: bool
    proof_rho: float        # rho  [0,1]; needed for PROCEED-leak / pass-rate metrics
    proof_term_id: str      # named theorem or diagnostic; needed for typed-diagnostic coverage


class NISPOrchestrator:
    """
    Wires the two-branch pipeline and drives one cycle per call.
    Only file that imports all layer implementations.

    Phase 1 cycle rate: 1 Hz. This matches TRIBE v2's natural output rate
    (1 Hz, decimated from the 2 Hz stimulus rate via adaptive average pooling).
    Sub-second cycles are a v2 silicon claim (Loihi 2 / Akida), not a v1 GPU claim.

    L1 is duck-typed across three implementations:
      LiveL1Transducer    real TRIBE v2 forward
      CachedL1Transducer  pre-computed file lookup
      TribevX             Gaussian fallback (always works)
    Required interface:
      async process(video_path, audio_path=None, text=None) -> CognitiveLatent

    L2 is duck-typed across three implementations:
      LiveL2Decoder       trained transformer with constrained beam search
      CachedL2Decoder     pre-computed file lookup
      StubL2Decoder       rule-based heuristic over Yeo-17 (always works)
    Required interface:
      async decode(latent) -> EDSLExpression

    L4 is duck-typed across three implementations:
      LiveL4Verifier      real Lean subprocess dispatch
      CachedL4Verifier    pre-computed file lookup
      StubL4Verifier      Python ontology mirror (always works)
    Required interface:
      async verify(edsl_expr, proof) -> ProofResult
    """

    def __init__(
        self,
        l1,                      # one of {LiveL1Transducer, CachedL1Transducer, TribevX}
        l2,                      # one of {LiveL2Decoder, CachedL2Decoder, StubL2Decoder}
        l3: L3proofSolver,
        l4,                      # one of {LiveL4Verifier, CachedL4Verifier, StubL4Verifier}
        l5: L5Arbiter,
        ig_monitor: IGMonitor,
        ece_monitor: ECEMonitor,
        clock: Callable[[], float],
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._l4 = l4
        self._l5 = l5
        self._ig = ig_monitor
        self._ece = ece_monitor
        self._clock = clock

    async def cycle(
        self,
        video_path: str | Path,
        proposed_action: Action,
        audio_path: str | Path | None = None,
        text: str | None = None,
    ) -> CycleResult:
        cycle_id = str(uuid.uuid4())
        t0 = self._clock()

        # Hypothesis branch (L1 -> L2) and ground-truth branch (L3 unconditional)
        # run concurrently. eDSL-term-driven L3 attention requires L2 to complete
        # first, breaking concurrency; unconditional polling is Phase 1's correct choice.
        (latent, edsl_expr), proof = await asyncio.gather(
            self._hypothesis_branch(video_path, audio_path, text),
            self._l3.poll_unconditional(),
        )

        proof = await self._l4.verify(edsl_expr, proof)
        verdict = self._l5.arbitrate(proof, proof, proposed_action)

        cert_root = generate_certificate(
            latent=latent,
            edsl_expr=edsl_expr,
            proof=proof,
            proof=proof,
            proposed_action=proposed_action,
            verdict=verdict,
            ece_window=self._ece.current_ece(),
            ig_window=self._ig.current_ig_ci95_lower(),
        )

        any_degraded = any([
            latent.degraded,
            edsl_expr.degraded,
            proof.degraded,
            proof.degraded,
            verdict.degraded,
        ])
        latency_ms = (self._clock() - t0) * 1_000

        log.info(
            "cycle_complete",
            extra={
                "cycle_id": cycle_id,
                "verdict": verdict.verdict.value,
                "cert_root": cert_root,
                "latency_ms": round(latency_ms, 2),
                "any_degraded": any_degraded,
                "proof_rho": proof.rho,
                "proof_term_id": proof.proof_term_id,
                "ig_ci95_lower": self._ig.current_ig_ci95_lower(),
                "ece": round(self._ece.current_ece(), 4),
            },
        )

        return CycleResult(
            cycle_id=cycle_id,
            verdict=verdict,
            certificate_root=cert_root,
            latency_ms=latency_ms,
            any_degraded=any_degraded,
            proof_rho=proof.rho,
            proof_term_id=proof.proof_term_id,
        )

    async def _hypothesis_branch(
        self,
        video_path: str | Path,
        audio_path: str | Path | None,
        text: str | None,
    ) -> tuple[CognitiveLatent, EDSLExpression]:
        latent = await self._l1.process(video_path, audio_path, text)
        edsl_expr = await self._l2.decode(latent)
        self._ece.observe(edsl_expr.confidence)
        return latent, edsl_expr
