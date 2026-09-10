from typing import List, Optional
import numpy as np


class MorphemeEmbeddingBuilder:
    def __init__(self, fasttext_model, segmenter, embedding_dim: int,
                 seed: int = 42, projection=None):
        self.fasttext_model = fasttext_model
        self.segmenter = segmenter
        self.embedding_dim = embedding_dim

        # W is square (Sec 4.1), so FastText and the model's embedding
        # space share a dimension. Checked here rather than at a shape error
        # deep in a forward pass, using the same check as build_projection.
        # Mismatched vectors are not reshaped or rectangularly projected.
        self.projection = projection
        if fasttext_model is not None:
            from .projection import check_dimensions
            check_dimensions(fasttext_model.get_dimension(), embedding_dim)

    def _fasttext_vec(self, text: str) -> Optional[np.ndarray]:
        if self.fasttext_model is None:
            return None
        try:
            vec = self.fasttext_model.get_word_vector(text)
        except Exception:
            return None
        if self.projection is not None:
            import torch
            device = next(self.projection.parameters()).device
            with torch.set_grad_enabled(self.projection.training):
                t = torch.as_tensor(vec, dtype=torch.float32, device=device)
                return self.projection(t)
        return vec

    def word_from_morphemes(self, morphemes: List[str]) -> Optional[np.ndarray]:
        """Average FastText vectors over an already-segmented morpheme list.
        Returns None (not a random vector) if none of the morphemes have a
        FastText vector; the caller applies the next step in the fallback
        chain: morphemes -> whole-token FastText -> character n-grams.
        """
        morpheme_vecs = [v for m in morphemes if (v := self._fasttext_vec(m)) is not None]
        if not morpheme_vecs:
            return None
        # These are torch tensors; np.mean would try to convert them and
        # raise "Can't call numpy() on Tensor that requires grad" whenever
        # they track gradients. Averaging in the tensor's own framework
        # handles both cases.
        import torch
        if isinstance(morpheme_vecs[0], torch.Tensor):
            return torch.stack(morpheme_vecs, dim=0).mean(dim=0)
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
