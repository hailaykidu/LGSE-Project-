from typing import Callable, Dict, Optional, Union
import torch


class LGSERegularizer:
    """Anchors new-token embeddings to their lexically grounded targets.

    The anchor can be supplied two ways:

      * a fixed tensor -- the paper's formulation, and the default;

      * a callable returning the anchor -- retained only for experiments
        that deliberately depart from the paper (see the note below).

    The paper (Sec 4.2) defines the penalty as

        L_reg = lambda * ||e_new - mu||^2

    "where mu is the initial embedding vector (e.g., from FastText
    projection)". mu is therefore a *constant*: the value the embedding was
    initialized to, frozen at that value, not recomputed as training
    proceeds. The term measures drift from initialization, which is what
    "prevent excessive deviation of new embeddings from their
    initialization" asks for.

    A live anchor recomputed through W would change what this term
    measures -- and would additionally give W a gradient path it does not
    have under the paper's formulation. That path is not part of the
    method; see DEVIATIONS.md section 1a.
    """

    def __init__(self,
                 init_embeddings: Union[torch.Tensor, Callable[[], torch.Tensor]],
                 token_ids: Dict[str, int],
                 lambda_reg: float,
                 device: str = "cpu"):
        self.lambda_reg = lambda_reg
        self.device = device
        self.token_id_list = list(token_ids.values())

        if callable(init_embeddings):
            self._anchor_fn: Optional[Callable[[], torch.Tensor]] = init_embeddings
            self.init_embeddings = None
        else:
            self._anchor_fn = None
            self.init_embeddings = init_embeddings.to(device)

    @property
    def anchor_is_live(self) -> bool:
        """True when the anchor is recomputed through W on every call."""
        return self._anchor_fn is not None

    def _anchor(self) -> torch.Tensor:
        if self._anchor_fn is not None:
            return self._anchor_fn().to(self.device)
        return self.init_embeddings

    def loss(self, embedding_layer: torch.nn.Embedding) -> torch.Tensor:
        if self.lambda_reg == 0.0:
            return torch.tensor(0.0, device=self.device)
        current = embedding_layer.weight[self.token_id_list]
        return self.lambda_reg * torch.mean((current - self._anchor()) ** 2)
