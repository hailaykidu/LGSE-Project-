from typing import List, Optional
import numpy as np


class MorphemeEmbeddingBuilder:
    def __init__(self, fasttext_model, segmenter, embedding_dim: int, seed: int = 42):
        self.fasttext_model = fasttext_model
        self.segmenter = segmenter
        self.embedding_dim = embedding_dim

        # FastText models are essentially always 300-dim (the standard
        # Facebook release / the fasttext.cc default), but the target
        # model's embedding space can be any width -- 768 for
        # xlm-roberta-base, for instance. Raw FastText vectors were being
        # written directly into the embedding matrix with no dimension
        # check at all, which crashes the moment the two widths differ (as
        # they did on the very first real run against xlm-roberta-base: a
        # 300 vs 768 mismatch). A fixed, seeded random projection maps
        # FastText-space vectors into the target space; per the
        # Johnson-Lindenstrauss lemma a random linear projection
        # approximately preserves relative distances, which is a
        # defensible way to bridge the two spaces without requiring extra
        # training data to fit a learned projection.
        self._projection = None
        if fasttext_model is not None:
            ft_dim = fasttext_model.get_dimension()
            if ft_dim != embedding_dim:
                rng = np.random.default_rng(seed)
                self._projection = rng.normal(
                    scale=1.0 / np.sqrt(ft_dim), size=(ft_dim, embedding_dim)
                ).astype(np.float32)

    def _fasttext_vec(self, text: str) -> Optional[np.ndarray]:
        if self.fasttext_model is None:
            return None
        try:
            vec = self.fasttext_model.get_word_vector(text)
        except Exception:
            return None
        if self._projection is not None:
            vec = vec @ self._projection
        return vec

    def word_from_morphemes(self, morphemes: List[str]) -> Optional[np.ndarray]:
        """Average FastText vectors over an already-segmented morpheme list.
        Returns None (not a random vector) if none of the morphemes have a
        FastText vector -- the caller is responsible for the next fallback
        step (character n-grams), matching the paper's described fallback
        chain: morphemes -> whole-token FastText -> character n-grams.
        """
        morpheme_vecs = [v for m in morphemes if (v := self._fasttext_vec(m)) is not None]
        if not morpheme_vecs:
            return None
        return np.mean(morpheme_vecs, axis=0)

    def build_embedding_for_token(self, token: str) -> Optional[np.ndarray]:
        """Convenience wrapper: segments the token itself, then tries
        morpheme-averaging, then whole-token FastText. Returns None (not a
        random vector) if neither works -- callers needing a guaranteed
        vector (e.g. no character n-gram encoder available) must supply
        their own final fallback.
        """
        morphemes = self.segmenter.segment(token)
        if morphemes:
            v = self.word_from_morphemes(morphemes)
            if v is not None:
                return v

        return self._fasttext_vec(token)
