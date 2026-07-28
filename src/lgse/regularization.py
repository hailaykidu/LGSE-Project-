from typing import Dict
import torch

class LGSERegularizer:
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
        return self.lambda_reg * torch.mean((current - self.init_embeddings) ** 2)
