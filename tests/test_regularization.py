import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lgse.regularization import LGSERegularizer


def test_loss_is_zero_when_embeddings_match_init():
    init = torch.zeros(3, 4)
    token_ids = {"a": 5, "b": 6, "c": 7}
    reg = LGSERegularizer(init_embeddings=init, token_ids=token_ids, lambda_reg=1.0)

    embedding = torch.nn.Embedding(10, 4)
    with torch.no_grad():
        embedding.weight[5:8] = init

    loss = reg.loss(embedding)
    assert loss.item() == 0.0


def test_loss_grows_with_deviation_from_init():
    init = torch.zeros(2, 4)
    token_ids = {"a": 5, "b": 6}
    reg = LGSERegularizer(init_embeddings=init, token_ids=token_ids, lambda_reg=1.0)

    embedding = torch.nn.Embedding(10, 4)
    with torch.no_grad():
        embedding.weight[5:7] = torch.ones(2, 4)  # deviates from init (all zeros) by 1.0 per dim

    loss = reg.loss(embedding)
    # mean squared deviation = mean of (1-0)^2 over all entries = 1.0
    assert abs(loss.item() - 1.0) < 1e-6


def test_lambda_zero_disables_regularization_regardless_of_deviation():
    init = torch.zeros(2, 4)
    token_ids = {"a": 5, "b": 6}
    reg = LGSERegularizer(init_embeddings=init, token_ids=token_ids, lambda_reg=0.0)

    embedding = torch.nn.Embedding(10, 4)
    with torch.no_grad():
        embedding.weight[5:7] = torch.full((2, 4), 100.0)  # huge deviation

    assert reg.loss(embedding).item() == 0.0


def test_only_specified_token_ids_are_penalized():
    # Deviating rows NOT in token_ids must not affect the loss at all.
    init = torch.zeros(1, 4)
    token_ids = {"a": 5}
    reg = LGSERegularizer(init_embeddings=init, token_ids=token_ids, lambda_reg=1.0)

    embedding = torch.nn.Embedding(10, 4)
    with torch.no_grad():
        embedding.weight[5] = torch.zeros(4)  # the regularized row: no deviation
        embedding.weight[6] = torch.full((4,), 999.0)  # unrelated row: huge deviation, ignored

    assert reg.loss(embedding).item() == 0.0
