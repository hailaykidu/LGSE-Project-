from typing import Dict
import torch


class LGSERegularizer:
    """Anchors new-token embeddings to their lexically grounded targets.

    Paper Sec 4.2:

        L_reg = lambda * ||e_new - mu||^2

    "where mu is the initial embedding vector (e.g., from FastText
    projection)". mu is a *constant* -- the value each embedding was
    initialized to, held fixed for the whole of training. The term therefore
    measures drift from initialization, which is what "prevent excessive
    deviation of new embeddings from their initialization" asks for.

    lambda is mandatory: the paper introduces it but never assigns a value.
    See LGSEConfig.reg_lambda and IMPLEMENTATION_NOTES.md section 8a.
    """

    def __init__(self,
                 init_embeddings: torch.Tensor,
                 token_ids: Dict[str, int],
                 lambda_reg: float,
                 device: str = "cpu"):
        self.lambda_reg = lambda_reg
        self.device = device
        self.token_id_list = list(token_ids.values())
        self.init_embeddings = init_embeddings.to(device)

    def loss(self, embedding_layer: torch.nn.Embedding) -> torch.Tensor:
        if self.lambda_reg == 0.0:
            return torch.tensor(0.0, device=self.device)
        current = embedding_layer.weight[self.token_id_list]
        return self.lambda_reg * torch.mean(
            (current - self.init_embeddings) ** 2)
