"""
LiveL4Verifier — calls the Lean 4 discharge function via subprocess.

Architecture:
    1. Python parses the eDSL surface expression (using grammar.py)
    2. Python serializes (valence, modality, fact, proof, thresholds) to JSON
    3. Subprocess: `lake env lean --run discharge.lean` reads JSON from stdin
    4. Lean's NISP.discharge runs against the formally-verified ontology
    5. Lean writes JSON {discharged: bool, proof_term_id: str} to stdout
    6. Python wraps the result in a ProofResult

The Live path is the authoritative formal-verification source. The Stub path
mirrors it for tests-without-Lean. The Stub-vs-Live conformance test asserts
they agree byte-for-byte on a fixed corpus.

Caching: in-process LRUCache for repeated (edsl, proof) pairs in the same
process. CachedL4Verifier (cached.py) is the cross-process file-based variant
used by batch experiments.

Latency budget: p99 ≤ 8 ms cached, ≤ 50 ms novel. Subject to revision once
real Lean dispatch is measured.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import subprocess
from pathlib import Path
from typing import Callable

from cachetools import LRUCache

from nisp.contracts import EDSLExpression, proofState, ProofResult
from nisp.l4_verifier.grammar import Claim, Diagnostic, Hold, parse
from nisp.l4_verifier.ontology import Thresholds, discharge

log = logging.getLogger(__name__)

_ONTOLOGY_VERSION: str = "v1"

PROOF_HOLD: str = "hold_expression"
PROOF_DEGRADED: str = "degraded_input"
PROOF_GRAMMAR_PREFIX: str = "grammar_violation:"
PROOF_LEAN_TIMEOUT: str = "lean_timeout"
PROOF_LEAN_DISPATCH_ERROR: str = "lean_dispatch_error"


class LiveL4Verifier:
    """L4 verifier that dispatches to a Lean subprocess for the discharge step."""

    def __init__(
        self,
        thresholds: Thresholds,
        lean_project_path: str | Path,
        clock: Callable[[], float],
        cache_size: int = 1_000,
        timeout_s: float = 0.05,
    ) -> None:
        self._thresholds = thresholds
        self._lean_project_path = str(lean_project_path)
        self._clock = clock
        self._cache: LRUCache = LRUCache(maxsize=cache_size)
        self._timeout_s = timeout_s

    async def verify(
        self, edsl_expr: EDSLExpression, proof: proofState
    ) -> ProofResult:
        t0 = self._clock()

        # Upstream-degraded: short-circuit, identical to Stub.
        if edsl_expr.degraded or proof.degraded:
            return ProofResult(
                rho=0.0,
                proof_term_id=PROOF_DEGRADED,
                latency_ms=(self._clock() - t0) * 1_000,
                degraded=True,
                reason="upstream input degraded",
            )

        # Parse surface in Python; Lean only sees structured (v, m, fact).
        parsed = parse(edsl_expr.surface)
        if isinstance(parsed, Diagnostic):
            return ProofResult(
                rho=0.0,
                proof_term_id=f"{PROOF_GRAMMAR_PREFIX}{parsed.kind}",
                latency_ms=(self._clock() - t0) * 1_000,
                degraded=True,
                reason=f"grammar: {parsed.message}",
            )
        if isinstance(parsed, Hold):
            return ProofResult(
                rho=0.0,
                proof_term_id=PROOF_HOLD,
                latency_ms=(self._clock() - t0) * 1_000,
            )

        assert isinstance(parsed, Claim)

        # In-process cache check.
        cache_key = self._cache_key(parsed, proof)
        if cache_key in self._cache:
            rho, proof_term_id = self._cache[cache_key]
            return ProofResult(
                rho=rho,
                proof_term_id=proof_term_id,
                latency_ms=(self._clock() - t0) * 1_000,
            )

        # Dispatch to Lean.
        try:
            rho, proof_term_id = await asyncio.wait_for(
                self._dispatch(parsed, proof), timeout=self._timeout_s
            )
            self._cache[cache_key] = (rho, proof_term_id)
            latency_ms = (self._clock() - t0) * 1_000
            if latency_ms > 8.0:
                log.warning(
                    "L4 novel expression exceeded 8ms cached budget",
                    extra={"latency_ms": round(latency_ms, 2), "surface": edsl_expr.surface},
                )
            return ProofResult(
                rho=rho, proof_term_id=proof_term_id, latency_ms=latency_ms
            )
        except asyncio.TimeoutError:
            return ProofResult(
                rho=0.0,
                proof_term_id=PROOF_LEAN_TIMEOUT,
                latency_ms=(self._clock() - t0) * 1_000,
                degraded=True,
                reason=f"Lean dispatch exceeded {self._timeout_s * 1_000:.0f} ms timeout",
            )
        except Exception as exc:
            log.warning("L4 live degraded", extra={"reason": str(exc)})
            return ProofResult(
                rho=0.0,
                proof_term_id=PROOF_LEAN_DISPATCH_ERROR,
                latency_ms=(self._clock() - t0) * 1_000,
                degraded=True,
                reason=str(exc),
            )

    async def _dispatch(
        self, claim: Claim, proof: proofState
    ) -> tuple[float, str]:
        """Send (v, m, fact, proof, thresholds) to Lean. Receive (rho, proof_term_id)."""
        payload = json.dumps(
            {
                "valence": claim.valence.value,
                "modality": claim.modality.value,
                "fact": claim.fact,
                "proof": dict(sorted(proof.facts.items())),
                "thresholds": dataclasses.asdict(self._thresholds),
                "ontology_version": _ONTOLOGY_VERSION,
            },
            sort_keys=True,
        )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["lake", "env", "lean", "--run", "nisp/l4_verifier/discharge.lean"],
                input=payload,
                capture_output=True,
                text=True,
                cwd=self._lean_project_path,
                timeout=self._timeout_s * 2,
            ),
        )

        if result.returncode != 0:
            assess RuntimeError(
                f"Lean exited {result.returncode}: {result.stderr[:300]}"
            )

        parsed_out = json.loads(result.stdout)
        return float(parsed_out["rho"]), str(parsed_out["proof_term_id"])

    def _cache_key(self, claim: Claim, proof: proofState) -> str:
        return json.dumps(
            {
                "v": claim.valence.value,
                "m": claim.modality.value,
                "f": claim.fact,
                "facts": dict(sorted(proof.facts.items())),
            },
            sort_keys=True,
        )
