import math
from typing import List, Dict, Any, Union
from app.services.relevance import RelevanceScorer
from app.services.decay import DecayEngine
from app.services.compressor import ContextCompressor
from app.services.injection_guard import InjectionGuard

class ContextorPipeline:
    """
    Main Contextor orchestrator integrating relevance scoring, time decay,
    compression of low-relevance items, and injection guarding for tool outputs.
    """

    def __init__(
        self,
        relevance_threshold: float = 0.40,
        drop_threshold: float = 0.15,
        decay_rate: float = 0.15,
        re_reference_threshold: float = 0.60
    ):
        self.relevance_threshold = relevance_threshold
        self.drop_threshold = drop_threshold
        self.scorer = RelevanceScorer()
        self.decay_engine = DecayEngine(decay_rate=decay_rate, re_reference_threshold=re_reference_threshold)
        self.compressor = ContextCompressor()
        self.guard = InjectionGuard()

    def estimate_tokens(self, text: str) -> int:
        """Approximate token count (1 token ~= 4 chars)."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4.0))

    def process(
        self,
        current_query: str,
        turn_history: List[Dict[str, Any]],
        current_turn: int
    ) -> Dict[str, Any]:
        """
        Process turn history and return optimized context payload + metrics.
        Each item in turn_history should be a dict with keys:
        'id', 'role', 'content', 'source', 'last_referenced_turn'
        """
        if not turn_history:
            return {
                "optimized_messages": [],
                "raw_tokens": 0,
                "contextor_tokens": 0,
                "token_reduction_pct": 0.0,
                "scores": [],
                "injection_flags": []
            }

        # 1. Injection Guard Scan on tool outputs
        injection_flags = []
        guarded_history = []

        for idx, item in enumerate(turn_history):
            role = item.get("role", "user")
            source = item.get("source", "agent_turn")
            content = item.get("content", "")

            item_copy = dict(item)

            if role == "tool" or source == "tool_output":
                scan_res = self.guard.scan(content, source="tool_output")
                if scan_res["is_flagged"]:
                    injection_flags.append({
                        "chunk_index": idx,
                        "reason": scan_res["reason"],
                        "confidence": scan_res["confidence"],
                        "original_snippet": content[:100]
                    })
                    item_copy["content"] = scan_res["sanitized_output"]
                    item_copy["is_flagged"] = True
                    item_copy["flag_reason"] = scan_res["reason"]

            guarded_history.append(item_copy)

        # Calculate raw token total
        raw_tokens = sum(self.estimate_tokens(item.get("content", "")) for item in guarded_history)

        # 2. Relevance Scoring
        raw_scores = self.scorer.compute_scores(query=current_query, chunks=guarded_history)

        # 3. Decay Processing
        last_turns = [item.get("last_referenced_turn", 0) for item in guarded_history]
        roles = [item.get("role", "user") for item in guarded_history]
        decay_results = self.decay_engine.process_batch(
            relevance_scores=raw_scores,
            last_referenced_turns=last_turns,
            current_turn=current_turn,
            roles=roles
        )

        # Combine scores
        detailed_scores = []
        for idx, item in enumerate(guarded_history):
            d_res = decay_results[idx]
            detailed_scores.append({
                "message_id": item.get("id"),
                "role": item.get("role"),
                "relevance_score": raw_scores[idx],
                "decay_weight": d_res["decay_weight"],
                "final_score": d_res["final_score"],
                "updated_last_referenced_turn": d_res["updated_last_referenced_turn"]
            })

        # 4. Filter & Compress
        high_relevance_messages = []
        low_relevance_chunks = []
        system_messages = []

        for idx, item in enumerate(guarded_history):
            role = item.get("role")
            score_info = decay_results[idx]
            final_score = score_info["final_score"]

            if role == "system":
                system_messages.append(item)
            elif final_score >= self.relevance_threshold:
                high_relevance_messages.append(item)
            elif final_score >= self.drop_threshold:
                low_relevance_chunks.append(item)

        # Compress low relevance bucket
        compressed_note = None
        if low_relevance_chunks:
            compressed_text = self.compressor.compress_chunks(low_relevance_chunks)
            compressed_note = {
                "role": "system",
                "content": compressed_text,
                "source": "compressed_summary",
                "is_compressed": True
            }

        # 5. Assemble optimized context array
        optimized = []
        optimized.extend(system_messages)
        if compressed_note:
            optimized.append(compressed_note)
        optimized.extend(high_relevance_messages)

        contextor_tokens = sum(self.estimate_tokens(item.get("content", "")) for item in optimized)

        reduction_pct = round(
            ((raw_tokens - contextor_tokens) / max(raw_tokens, 1)) * 100.0, 2
        ) if raw_tokens > 0 else 0.0

        return {
            "optimized_messages": optimized,
            "raw_tokens": raw_tokens,
            "contextor_tokens": contextor_tokens,
            "token_reduction_pct": reduction_pct,
            "scores": detailed_scores,
            "injection_flags": injection_flags
        }
