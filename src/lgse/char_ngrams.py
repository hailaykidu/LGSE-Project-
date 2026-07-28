from typing import List
import torch
from collections import defaultdict

class CharNgramEncoder:
    def __init__(self, n_min: int, n_max: int, dim: int, device: str = "cpu"):
        self.n_min = n_min
        self.n_max = n_max
        self.dim = dim
        self.device = device
        self.ngram_table = defaultdict(self._init_vec)

    def _init_vec(self):
        return torch.randn(self.dim, device=self.device) * 0.02

    def _ngrams(self, token: str) -> List[str]:
        token = f"<{token}>"
        ngrams: List[str] = []
        for n in range(self.n_min, self.n_max + 1):
            for i in range(len(token) - n + 1):
                ngrams.append(token[i:i+n])
        return ngrams

    def encode(self, token: str) -> torch.Tensor:
        ngrams = self._ngrams(token)
        vecs = [self.ngram_table[ng] for ng in ngrams]
        return torch.stack(vecs, dim=0).mean(dim=0)
