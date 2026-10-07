import math
from typing import List, Dict, Any, Union

class DecayEngine:
    """
    Applies time/turn-based exponential decay to relevance scores.
    Formula:
      decay_factor = exp(-decay_rate * (current_turn - last_referenced_turn))
      final_score = relevance_score * decay_factor

    If a chunk's relevance_score exceeds the re_reference_threshold,
    its last_referenced_turn is updated to current_turn, resetting decay.
    """

    def __init__(self, decay_rate: float = 0.15, re_reference_threshold: float = 0.6):
        self.decay_rate = decay_rate
        self.re_reference_threshold = re_reference_threshold

    def calculate_decay(
        self,
        relevance_score: float,
        last_referenced_turn: int,
        current_turn: int,
        is_system_role: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate decayed score for a single chunk.
        System role chunks do not decay.
        """
        if is_system_role:
            return {
                "decay_weight": 1.0,
                "final_score": relevance_score,
                "updated_last_referenced_turn": current_turn,
                "re_referenced": True
            }

        # Check if chunk is re-referenced in current turn
        re_referenced = relevance_score >= self.re_reference_threshold
        updated_turn = current_turn if re_referenced else last_referenced_turn

        turn_delta = max(0, current_turn - updated_turn)
        decay_weight = math.exp(-self.decay_rate * turn_delta)
        final_score = round(max(0.0, min(1.0, relevance_score * decay_weight)), 4)

        return {
            "decay_weight": round(decay_weight, 4),
            "final_score": final_score,
            "updated_last_referenced_turn": updated_turn,
            "re_referenced": re_referenced
        }

    def process_batch(
        self,
        relevance_scores: List[float],
        last_referenced_turns: List[int],
        current_turn: int,
        roles: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Process a list of chunk scores at current_turn."""
        results = []
        if roles is None:
            roles = ["user"] * len(relevance_scores)

        for rel_score, last_turn, role in zip(relevance_scores, last_referenced_turns, roles):
            res = self.calculate_decay(
                relevance_score=rel_score,
                last_referenced_turn=last_turn,
                current_turn=current_turn,
                is_system_role=(role == "system")
            )
            results.append(res)

        return results
