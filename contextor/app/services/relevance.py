import math
import re
from typing import List, Dict, Any, Union

class RelevanceScorer:
    """
    Computes relevance scores in [0.0, 1.0] between a turn query and historical context chunks.
    Uses term frequency-inverse document frequency (TF-IDF) + cosine vector similarity,
    with keyword matching fallback to ensure zero external dependency failure.
    """

    def __init__(self, system_role_min_score: float = 1.0):
        self.system_role_min_score = system_role_min_score

    def tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer with lowercasing."""
        return re.findall(r'\w+', text.lower())

    def compute_scores(self, query: str, chunks: List[Union[str, Dict[str, Any]]]) -> List[float]:
        """
        Compute normalized relevance score for each chunk relative to query.
        Accepts list of raw strings or dicts with 'content' and 'role' keys.
        """
        if not chunks:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return [1.0] * len(chunks)

        # Extract text content and roles
        text_list = []
        roles = []
        for item in chunks:
            if isinstance(item, dict):
                text_list.append(item.get("content", ""))
                roles.append(item.get("role", "user"))
            elif hasattr(item, "content"):
                text_list.append(getattr(item, "content", ""))
                roles.append(getattr(item, "role", "user"))
            else:
                text_list.append(str(item))
                roles.append("user")

        # Build vocabulary across query + documents
        doc_tokens_list = [self.tokenize(t) for t in text_list]
        all_docs = [query_tokens] + doc_tokens_list
        vocab = sorted(list(set(token for doc in all_docs for token in doc)))

        if not vocab:
            return [1.0] * len(chunks)

        vocab_idx = {word: i for i, word in enumerate(vocab)}

        # Document Frequency (DF)
        num_docs = len(all_docs)
        df: Dict[str, int] = {}
        for doc in all_docs:
            unique_words = set(doc)
            for word in unique_words:
                df[word] = df.get(word, 0) + 1

        # Inverse Document Frequency (IDF)
        idf = {word: math.log((num_docs + 1) / (df.get(word, 0) + 1)) + 1.0 for word in vocab}

        def get_tfidf_vec(doc_tokens: List[str]) -> List[float]:
            tf: Dict[str, int] = {}
            for w in doc_tokens:
                tf[w] = tf.get(w, 0) + 1
            vec = [0.0] * len(vocab)
            total = len(doc_tokens) if doc_tokens else 1
            for w, count in tf.items():
                if w in vocab_idx:
                    vec[vocab_idx[w]] = (count / total) * idf[w]
            return vec

        def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(a * a for a in vec_a))
            norm_b = math.sqrt(sum(b * b for b in vec_b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        query_vec = get_tfidf_vec(query_tokens)
        scores: List[float] = []

        for idx, doc_tokens in enumerate(doc_tokens_list):
            role = roles[idx]
            if role == "system":
                scores.append(self.system_role_min_score)
                continue

            doc_vec = get_tfidf_vec(doc_tokens)
            cos_sim = cosine_similarity(query_vec, doc_vec)

            # Add keyword overlap boost for exact token matches
            overlap = len(set(query_tokens).intersection(set(doc_tokens)))
            overlap_ratio = overlap / max(len(set(query_tokens)), 1)
            
            composite = 0.7 * cos_sim + 0.3 * overlap_ratio
            # Clamp between 0.1 and 1.0 (minimum base score 0.1 so history isn't completely zeroed out unless decayed)
            final_val = round(max(0.1, min(1.0, composite)), 4)
            scores.append(final_val)

        return scores
