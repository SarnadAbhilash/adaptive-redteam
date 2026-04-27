"""
adaptive-redteam: adaptive red-teaming engine for discovering LLM failure modes.

Layer 2 of the 3-repo AI safety research stack. Depends on safety-probe.
"""

from adaptive_redteam.categories.hallucination import HallucinationCategory
from adaptive_redteam.categories.instruction_hierarchy import InstructionHierarchyCategory
from adaptive_redteam.categories.multi_turn_consistency import MultiTurnConsistencyCategory
from adaptive_redteam.categories.overconfidence import OverconfidenceCategory
from adaptive_redteam.categories.sycophancy import SycophancyCategory
from adaptive_redteam.failure_mode import FailureMode
from adaptive_redteam.mutators.escalate import EscalateMutator
from adaptive_redteam.mutators.paraphrase import ParaphraseMutator
from adaptive_redteam.mutators.reframe import ReframeMutator
from adaptive_redteam.schemas.score import RedTeamScore
from adaptive_redteam.schemas.seed import Seed
from adaptive_redteam.scorers.hallucination_scorer import HallucinationScorer
from adaptive_redteam.scorers.instruction_hierarchy_scorer import InstructionHierarchyScorer
from adaptive_redteam.scorers.multi_turn_consistency_scorer import MultiTurnConsistencyScorer
from adaptive_redteam.scorers.overconfidence_scorer import OverconfidenceScorer
from adaptive_redteam.scorers.sycophancy_scorer import SycophancyScorer

__version__ = "0.1.0"
__all__ = [
    "FailureMode",
    "Seed",
    "RedTeamScore",
    "SycophancyCategory",
    "HallucinationCategory",
    "InstructionHierarchyCategory",
    "OverconfidenceCategory",
    "MultiTurnConsistencyCategory",
    "ParaphraseMutator",
    "EscalateMutator",
    "ReframeMutator",
    "SycophancyScorer",
    "HallucinationScorer",
    "InstructionHierarchyScorer",
    "OverconfidenceScorer",
    "MultiTurnConsistencyScorer",
]
