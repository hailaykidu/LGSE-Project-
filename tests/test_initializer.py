import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lgse.char_ngrams import CharNgramEncoder
from lgse.initializer import LGSEInitializer
from lgse.morpheme_embeddings import MorphemeEmbeddingBuilder
from lgse.segmentation import MorphologicalSegmenter


class FakeFastText:
    def __init__(self, vectors: dict):
        self.vectors = vectors

    def get_dimension(self):
        return 4

    def get_word_vector(self, text):
        if text in self.vectors:
            return np.array(self.vectors[text], dtype=np.float32)
        raise KeyError(text)


def make_initializer(embedding_layer, segmenter, ft_vectors):
    morph_builder = MorphemeEmbeddingBuilder(
        fasttext_model=FakeFastText(ft_vectors), segmenter=segmenter, embedding_dim=4
    )
    char_encoder = CharNgramEncoder(n_min=2, n_max=3, dim=4)
    return LGSEInitializer(
        embedding_layer=embedding_layer, morph_builder=morph_builder, char_encoder=char_encoder
    )


def test_falls_back_through_morpheme_then_whole_token_then_char_ngrams():
    embedding = torch.nn.Embedding(5, 4)
    seg = MorphologicalSegmenter({"ሰላማዊ": ["ሰላም", "ኣዊ"]})

    init = make_initializer(embedding, seg, ft_vectors={
        "ሰላም": [1.0, 0.0, 0.0, 0.0],
        "ኣዊ": [0.0, 1.0, 0.0, 0.0],
        "ኮምፒተር": [0.0, 0.0, 1.0, 0.0],
    })

    # 1) known morphemes -> averaged FastText vector
    v1 = init.init_token_embedding("ሰላማዊ")
    assert isinstance(v1, torch.Tensor)
    torch.testing.assert_close(v1, torch.tensor([0.5, 0.5, 0.0, 0.0]))

    # 2) no segmentation, but whole-token FastText hit
    v2 = init.init_token_embedding("ኮምፒተር")
    torch.testing.assert_close(v2, torch.tensor([0.0, 0.0, 1.0, 0.0]))

    # 3) neither -> character n-gram fallback (deterministic given a fixed
    # CharNgramEncoder instance; just check it's a real 4-dim tensor, not
    # a crash and not accidentally identical to the other two cases)
    v3 = init.init_token_embedding("ፈጺሙ_ዘይፍለጥ")
    assert isinstance(v3, torch.Tensor)
    assert v3.shape == (4,)
    assert not torch.allclose(v3, v1)
    assert not torch.allclose(v3, v2)


def test_write_embeddings_for_new_tokens_writes_correct_rows_in_place():
    embedding = torch.nn.Embedding(5, 4)
    original_weight_ptr = embedding.weight.data_ptr()
    seg = MorphologicalSegmenter({"ሰላማዊ": ["ሰላም", "ኣዊ"]})
    init = make_initializer(embedding, seg, ft_vectors={
        "ሰላም": [1.0, 0.0, 0.0, 0.0], "ኣዊ": [0.0, 1.0, 0.0, 0.0],
    })

    token_to_id = {"ሰላማዊ": 3}
    init_matrix = init.write_embeddings_for_new_tokens(token_to_id)

    torch.testing.assert_close(embedding.weight.data[3], torch.tensor([0.5, 0.5, 0.0, 0.0]))
    torch.testing.assert_close(init_matrix[0], torch.tensor([0.5, 0.5, 0.0, 0.0]))

    # Written in place (.data indexing), not via a Parameter replacement --
    # this is what keeps a tied MLM output head in sync (see the
    # LGSEInitializer docstring for why a full Parameter swap breaks tying).
    assert embedding.weight.data_ptr() == original_weight_ptr
