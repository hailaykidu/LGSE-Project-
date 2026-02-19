from typing import List, Optional
import numpy as np

class MorphemeEmbeddingBuilder:
    def __init__(self, fasttext_model, segmenter, embedding_dim: int):
        self.fasttext_model = fasttext_model
        self.segmenter = segmenter
        self.embedding_dim = embedding_dim

    def _fasttext_vec(self, text: str) -> Optional[np.ndarray]:
        if self.fasttext_model is None:
            return None
        try:
            return self.fasttext_model.get_word_vector(text)
        except Exception:
            return None

    def build_embedding_for_token(self, token: str) -> np.ndarray:
        # 1) try morphemes from lexicon
        morphemes: List[str] = self.segmenter.segment(token)

        morpheme_vecs: List[np.ndarray] = []
        for m in morphemes:
            v = self._fasttext_vec(m)
            if v is not None:
                morpheme_vecs.append(v)

        if morpheme_vecs:
            return np.mean(morpheme_vecs, axis=0)

        # 2) try FastText on the whole token
        v_token = self._fasttext_vec(token)
        if v_token is not None:
            return v_token

        # 3) final fallback: random small vector
        return np.random.normal(scale=0.02, size=(self.embedding_dim,))
