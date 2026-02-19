from typing import List, Dict, Tuple
import torch

from .segmentation import MorphologicalSegmenter
from .morpheme_embeddings import MorphemeEmbeddingBuilder
from .char_ngrams import CharNgramEncoder

class LGSEInitializer:
    def __init__(self,
                 tokenizer,
                 embedding_layer: torch.nn.Embedding,
                 segmenter: MorphologicalSegmenter,
                 morph_builder: MorphemeEmbeddingBuilder,
                 char_encoder: CharNgramEncoder,
                 device: str = "cpu"):
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.segmenter = segmenter
        self.morph_builder = morph_builder
        self.char_encoder = char_encoder
        self.device = device

    def init_token_embedding(self, token: str) -> torch.Tensor:
        morphemes = self.segmenter.segment(token)
        if morphemes:
            v = self.morph_builder.word_from_morphemes(morphemes)
            if v is not None:
                return v
        return self.char_encoder.encode(token)

    def add_new_tokens(self, new_tokens: List[str]) -> Tuple[Dict[str, int], torch.Tensor]:
        added = self.tokenizer.add_tokens(new_tokens)
        if added != len(new_tokens):
            raise ValueError("Tokenizer did not add all new tokens.")

        old_weight = self.embedding_layer.weight.data
        old_num, dim = old_weight.shape
        new_weight = torch.zeros(old_num + added, dim, device=self.device, dtype=old_weight.dtype)
        new_weight[:old_num] = old_weight

        init_vecs = []
        for tok in new_tokens:
            v = self.init_token_embedding(tok)
            init_vecs.append(v)
        init_matrix = torch.stack(init_vecs, dim=0)
        new_weight[old_num:] = init_matrix

        self.embedding_layer.weight = torch.nn.Parameter(new_weight)

        token_to_id: Dict[str, int] = {
            tok: self.tokenizer.convert_tokens_to_ids(tok) for tok in new_tokens
        }
        return token_to_id, init_matrix.detach().clone()
