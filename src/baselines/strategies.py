"""
strategies.py -- new-token initialization baselines for Table 2.

Table 2 compares five systems. They differ only in how the embeddings of
newly added vocabulary tokens are initialized before LAPT; the LAPT
procedure itself is identical across all of them.

    XLM-R            no vocabulary expansion, no LAPT
    +LAPT            vocabulary expanded, new rows left as the model's own
                     default initialization
    +Random+LAPT     new rows drawn from a Gaussian
    +FOCUS+LAPT      new rows as a similarity-weighted combination of
                     existing embeddings (Dobler & de Melo, 2023)
    +LGSE+LAPT       morpheme -> FastText -> char n-gram chain (this paper)

Each strategy exposes the same interface as LGSEInitializer so the trainer
can swap between them with no other change.
"""

from typing import Dict

import torch


class BaselineInitializer:
    """Shared interface: fill rows of an embedding matrix for new tokens."""

    name = "base"

    def __init__(self, embedding_layer: torch.nn.Embedding, device: str = "cpu"):
        self.embedding_layer = embedding_layer
        self.device = device

    def init_token_embedding(self, token: str) -> torch.Tensor:
        raise NotImplementedError

    def write_embeddings_for_new_tokens(
            self, token_to_id: Dict[str, int]) -> torch.Tensor:
        tokens = list(token_to_id.keys())
        vecs = [self.init_token_embedding(tok) for tok in tokens]
        init_matrix = torch.stack(vecs, dim=0).to(
            dtype=self.embedding_layer.weight.dtype,
            device=self.embedding_layer.weight.device,
        )
        with torch.no_grad():
            for tok, vec in zip(tokens, init_matrix):
                self.embedding_layer.weight.data[token_to_id[tok]] = vec
        return init_matrix.detach().clone()


class DefaultInit(BaselineInitializer):
    """+LAPT: leave whatever resize_token_embeddings() produced."""

    name = "default"

    def write_embeddings_for_new_tokens(
            self, token_to_id: Dict[str, int]) -> torch.Tensor:
        ids = list(token_to_id.values())
        return self.embedding_layer.weight.data[ids].detach().clone()


class RandomInit(BaselineInitializer):
    """+Random+LAPT: Gaussian at the scale of the existing embeddings."""

    name = "random"

    def __init__(self, embedding_layer, device="cpu", seed: int = 42,
                 old_vocab_size: int = None):
        super().__init__(embedding_layer, device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        # Match the std of the pretrained rows rather than assuming 0.02,
        # so the baseline is not handicapped by an arbitrary scale.
        rows = embedding_layer.weight.data[:old_vocab_size] \
            if old_vocab_size else embedding_layer.weight.data
        self.std = float(rows.std())

    def init_token_embedding(self, token: str) -> torch.Tensor:
        dim = self.embedding_layer.embedding_dim
        v = torch.randn(dim, generator=self.generator) * self.std
        return v.to(self.device)


class FocusInit(BaselineInitializer):
    """+FOCUS+LAPT: similarity-weighted combination of existing embeddings.

    FOCUS (Dobler & de Melo, 2023) initializes each new token as a convex
    combination of the pretrained embeddings of overlapping tokens, weighted
    by similarity in an auxiliary space. Here the auxiliary space is the
    FastText space already loaded for LGSE, which keeps the two methods
    comparable: they see the same external signal and differ only in how
    they use it.

    Falls back to the mean pretrained embedding when a token has no
    auxiliary vector, rather than to a random draw, so the baseline stays
    deterministic.
    """

    name = "focus"

    def __init__(self, embedding_layer, aux_vectors, aux_index,
                 device="cpu", top_k: int = 10, old_vocab_size: int = None):
        super().__init__(embedding_layer, device)
        self.aux_vectors = aux_vectors          # (V_old, d_aux), normalized
        self.aux_index = aux_index              # row -> embedding-matrix id
        self.top_k = top_k
        self.old_vocab_size = old_vocab_size
        self._mean = embedding_layer.weight.data[:old_vocab_size].mean(0) \
            if old_vocab_size else embedding_layer.weight.data.mean(0)
        self._aux_lookup = None

    def set_aux_lookup(self, fn):
        """fn(token) -> vector in the auxiliary space, or None."""
        self._aux_lookup = fn

    def init_token_embedding(self, token: str) -> torch.Tensor:
        if self._aux_lookup is None or self.aux_vectors is None:
            return self._mean.to(self.device)
        q = self._aux_lookup(token)
        if q is None:
            return self._mean.to(self.device)

        q = torch.as_tensor(q, dtype=torch.float32)
        q = q / (q.norm() + 1e-8)
        sims = self.aux_vectors @ q
        k = min(self.top_k, sims.numel())
        top = torch.topk(sims, k)
        weights = torch.softmax(top.values, dim=0)
        rows = [self.aux_index[i] for i in top.indices.tolist()]
        source = self.embedding_layer.weight.data[rows]
        return (weights.unsqueeze(1) * source).sum(0).to(self.device)


INITIALIZERS = {
    "default": DefaultInit,
    "random": RandomInit,
    "focus": FocusInit,
}


def build_initializer(kind: str, **kwargs):
    if kind not in INITIALIZERS:
        raise ValueError(f"unknown initializer {kind!r}; "
                         f"expected one of {sorted(INITIALIZERS)} or 'lgse'")
    return INITIALIZERS[kind](**kwargs)
