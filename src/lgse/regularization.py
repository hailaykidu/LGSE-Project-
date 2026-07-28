from typing import Callable, Dict, Optional, Union
import torch


class LGSERegularizer:
    """Anchors new-token embeddings to their lexically grounded targets.

    The anchor can be supplied two ways:

      * a fixed tensor -- used by the baselines, which have no projection,
        and by any run where the target is a constant;

      * a callable returning the anchor -- used by LGSE proper, where the
        target is W(f) and must stay a live function of W.

    The distinction matters. With a detached anchor the penalty
    lambda*||E_new - anchor||^2 is a function of the embeddings alone, so
    W receives no gradient from it: the initializer writes W's output into
    the embedding through `.data`, which severs the graph, and nothing
    downstream depends on W again. W would then sit in the optimizer,
    report as trainable, and never move -- reaching exactly the same end
    state as a frozen random map.

    Recomputing the anchor through W each step is what makes "trained
    jointly with the new embeddings" true: both sides of the penalty are
    live, so the term pulls the embeddings toward W(f) and simultaneously
    adapts W toward the embeddings the LM is learning.
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
