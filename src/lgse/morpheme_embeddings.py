from typing import List, Optional
import numpy as np


class MorphemeEmbeddingBuilder:
    def __init__(self, fasttext_model, segmenter, embedding_dim: int,
                 seed: int = 42, projection=None):
        self.fasttext_model = fasttext_model
        self.segmenter = segmenter
        self.embedding_dim = embedding_dim

        # FastText is 300-dim; the target embedding space is wider (768 for
        # xlm-roberta-base). The learned projection W bridges the two -- see
        # src/lgse/projection.py. It is held here and applied in
        # _fasttext_vec; the trainer registers its parameters with the
        # optimizer so gradients actually reach it.
        #
        # There is deliberately no fallback for a missing projection under a
        # dimension mismatch. Substituting an untrained map here would leave
        # the run reporting as LGSE while the projection never learns
        # anything, so this raises instead.
        self.projection = projection
        if projection is None and fasttext_model is not None:
            ft_dim = fasttext_model.get_dimension()
            if ft_dim != embedding_dim:
                raise ValueError(
                    f"FastText is {ft_dim}-dim but the embedding space is "
                    f"{embedding_dim}-dim, and no learned projection was "
                    f"supplied. Build one with "
                    f"lgse.projection.build_projection({ft_dim}, "
                    f"{embedding_dim}) and pass it as `projection=`.")

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
        # With a learned projection these are torch tensors carrying
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
