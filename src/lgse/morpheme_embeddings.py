from typing import List, Optional
import numpy as np


class MorphemeEmbeddingBuilder:
    def __init__(self, fasttext_model, segmenter, embedding_dim: int,
                 seed: int = 42, projection=None):
        self.fasttext_model = fasttext_model
        self.segmenter = segmenter
        self.embedding_dim = embedding_dim

        # LGSE's W is square (paper Sec 4.1), so FastText and the model's
        # embedding space must share a dimension. Verify that here rather
        # than waiting for a shape error deep in a forward pass, and use the
        # same check as build_projection so the two cannot disagree.
        #
        # There is deliberately no fallback: reshaping or rectangularly
        # projecting mismatched vectors would leave the run reporting as
        # LGSE while computing something the paper does not describe.
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
        FastText vector -- the caller is responsible for the next fallback
        step (character n-grams), matching the paper's described fallback
        chain: morphemes -> whole-token FastText -> character n-grams.
        """
        morpheme_vecs = [v for m in morphemes if (v := self._fasttext_vec(m)) is not None]
        if not morpheme_vecs:
            return None
        # With a trainable W these are torch tensors carrying
        # gradients, and np.mean would try to convert them -- which raises
        # "Can't call numpy() on Tensor that requires grad". Averaging in the
        # tensor's own framework keeps W on the autograd graph, which is the
        # whole point of learning it.
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
