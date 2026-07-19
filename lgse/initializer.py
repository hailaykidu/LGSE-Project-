from typing import Dict
import torch

from .morpheme_embeddings import MorphemeEmbeddingBuilder
from .char_ngrams import CharNgramEncoder


class LGSEInitializer:
    """
    Computes and writes initialization vectors for newly added tokens.

    Deliberately does NOT resize the embedding matrix itself -- that is the
    caller's responsibility via the tokenizer's own `add_tokens()` and the
    model's own `resize_token_embeddings()`. An earlier version of this
    class reconstructed the weight matrix by hand and replaced
    `embedding_layer.weight` with a new Parameter, which silently breaks
    weight tying between the input embeddings and a tied MLM output head
    (common for XLM-R-style models): the tied head would keep pointing at
    the old, smaller Parameter object instead of following the resize.
    Using the model's own resize API first keeps both sides of the tie
    consistent; this class only fills in the *values* for the new rows.
    """

    def __init__(self,
                 embedding_layer: torch.nn.Embedding,
                 morph_builder: MorphemeEmbeddingBuilder,
                 char_encoder: CharNgramEncoder,
                 device: str = "cpu"):
        self.embedding_layer = embedding_layer
        self.morph_builder = morph_builder
        self.char_encoder = char_encoder
        self.device = device

    def init_token_embedding(self, token: str) -> torch.Tensor:
        """Morpheme-average -> whole-token FastText -> character n-grams,
        matching the paper's described fallback chain. Always returns a
        torch.Tensor (not a mix of numpy arrays and tensors) so callers can
        safely torch.stack() the results.
        """
        v = self.morph_builder.build_embedding_for_token(token)
        if v is not None:
            return torch.as_tensor(v, dtype=torch.float32, device=self.device)

        return self.char_encoder.encode(token).to(self.device)

    def write_embeddings_for_new_tokens(self, token_to_id: Dict[str, int]) -> torch.Tensor:
        """Writes an init vector into `embedding_layer` for each token in
        `token_to_id` (in place, via .data indexing -- not a Parameter
        replacement, so weight tying with any tied output head survives).
        Returns the stacked init matrix (same order as dict iteration) so
        the caller can hand it to LGSERegularizer as the anchor to
        regularize toward.
        """
        tokens = list(token_to_id.keys())
        vecs = [self.init_token_embedding(tok) for tok in tokens]
        init_matrix = torch.stack(vecs, dim=0).to(
            dtype=self.embedding_layer.weight.dtype, device=self.embedding_layer.weight.device
        )

        with torch.no_grad():
            for tok, vec in zip(tokens, init_matrix):
                idx = token_to_id[tok]
                self.embedding_layer.weight.data[idx] = vec

        return init_matrix.detach().clone()
