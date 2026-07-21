"""
    (hold)
"""
from __future__ import annotations

import enum
from typing import Final


# - Enums -

class Valence(enum.Enum):
    DETECT = "detect"         # Scene understanding (e.g., UCF)
    COMMAND = "command"       # Robot control (e.g., Cobot)
    HESITATE = "hesitate"

class Modality(str, enum.Enum):
    BELIEF = "belief"
    DESIRE = "desire"
    ATTENTION = "attention"
    SOCIAL = "social"
    ANTICIPATE = "anticipate"

_SCENE_FACTS: Final[frozenset[str]] = frozenset({
    "anomaly", "normal", "proceed", "abort",
    "approaching", "retreating", "gazing_at", "pointing",
    "interacting", "isolated"
})

ESCALATION_VALENCES: Final[frozenset[Valence]] = frozenset({
    Valence.DETECT,
    Valence.COMMAND,
})

# - Unit Vocabulary -

UNIT_OPEN: Final[str] = "("
TOK_CLOSE: Final[str] = ")"
TOK_EOS: Final[str] = "<eos>"
TOK_HOLD: Final[str] = "hold"
TOK_DETECT: Final[str] = "detect"
TOK_COMMAND: Final[str] = "command"
TOK_HESITATE: Final[str] = "hesitate"
TOK_VALENCE: Final[frozenset[str]] = frozenset(v.value for v in Valence)
TOK_MODALITY: Final[frozenset[str]] = frozenset(m.value for m in Modality)
TOK_FACTS: Final[frozenset[str]] = frozenset(_SCENE_FACTS)

# Total ordered vocabulary (order is stable - at no point reorder after training)
VOCAB: Final[tuple[str, ...]] = (
    UNIT_OPEN,
    TOK_CLOSE,
    TOK_EOS,
    TOK_HOLD,
    TOK_DETECT,
    TOK_COMMAND,
    TOK_HESITATE,
    *sorted(TOK_MODALITY),
    *sorted(TOK_FACTS),
)

VOCAB_SIZE: Final[int] = len(VOCAB)
UNIT_TO_IDX: Final[dict[str, int]] = {t: i for i, t in enumerate(VOCAB)}
