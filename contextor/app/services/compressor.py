import os
import re
from typing import List, Dict, Any, Union

class ContextCompressor:
    """
    Summarizes low-relevance context chunks into concise condensed notes.
    Uses LiteLLM / OpenAI API if available, with a fast extractive summary fallback.
    """

    def __init__(self, summary_model: str = "gpt-3.5-turbo"):
        self.summary_model = summary_model

    def compress_chunks(self, chunks: List[Union[str, Dict[str, Any]]]) -> str:
        """
        Compress multiple text chunks into a single condensed summary note.
        """
        if not chunks:
            return ""

        extracted_texts = []
        for c in chunks:
            if isinstance(c, dict):
                role = c.get("role", "context")
                content = c.get("content", "")
                extracted_texts.append(f"[{role}]: {content}")
            elif hasattr(c, "content"):
                role = getattr(c, "role", "context")
                content = getattr(c, "content", "")
                extracted_texts.append(f"[{role}]: {content}")
            else:
                extracted_texts.append(str(c))

        full_raw_text = "\n".join(extracted_texts)

        # Attempt LiteLLM call if API key present
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                import litellm
                response = litellm.completion(
                    model=self.summary_model,
                    messages=[
                        {"role": "system", "content": "You are a concise context compressor. Summarize historical turn interactions into a 2-3 sentence key point summary capturing facts, parameters, and results."},
                        {"role": "user", "content": f"Summarize these turns:\n{full_raw_text}"}
                    ],
                    max_tokens=150
                )
                summary = response.choices[0].message.content.strip()
                return f"[Compressed History Note]: {summary}"
            except Exception:
                pass  # Fallback to local extractive summarizer

        return self._extractive_fallback(extracted_texts)

    def _extractive_fallback(self, extracted_texts: List[str]) -> str:
        """
        Fast local extractive summarizer that extracts key statements and tool results.
        """
        summary_lines = []
        for line in extracted_texts:
            # Extract first sentence or clean payload
            clean = line.strip()
            # Remove repeated whitespace
            clean = re.sub(r'\s+', ' ', clean)
            if len(clean) > 120:
                clean = clean[:117] + "..."
            summary_lines.append(clean)

        combined = " | ".join(summary_lines[:5])
        if len(summary_lines) > 5:
            combined += f" (...and {len(summary_lines) - 5} older turns)"
        return f"[Compressed History Note]: {combined}"
