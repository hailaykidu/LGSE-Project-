import os
import random

import fasttext
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

from .char_ngrams import CharNgramEncoder
from .config import LGSEConfig
from .initializer import LGSEInitializer
from .morpheme_embeddings import MorphemeEmbeddingBuilder
from .regularization import LGSERegularizer
from .segmentation import MorphologicalSegmenter
from .token_selection import load_new_tokens


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LGSELAPTrainer:
    """
    Language-Adaptive Pretraining with LGSE-initialized new-token embeddings.

    Implements the specific claims from the LGSE paper that a previous
    version of this codebase defined supporting classes for but never
    actually wired together or ran:
      - new tokens get morpheme-averaged / FastText / character-n-gram
        embeddings (via LGSEInitializer), not random vectors
      - a regularization term anchors new embeddings to their init values
        during training (LGSERegularizer, applied every step below)
      - the base model is frozen and only the new-token embedding rows are
        updated, isolating the effect of the initialization strategy
    """

    def __init__(self, config: LGSEConfig, dataset):
        set_seed(config.seed)
        self.config = config

        requested_device = config.device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[LGSELAPTrainer] '{requested_device}' requested but no CUDA device "
                  f"is available -- falling back to CPU.")
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        # 1) base model + tokenizer. AutoModelForMaskedLM (not the bare
        # AutoModel encoder a previous version used) is required for there
        # to be an actual MLM loss/head to train against at all.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(config.model_name).to(self.device)

        # 2) lexicon + FastText
        segmenter = MorphologicalSegmenter.from_file(config.morph_lexicon_path)
        print(f"Loaded morphological lexicon: {len(segmenter.lexicon)} words")
        ft_model = fasttext.load_model(config.fasttext_path)

        # 3) add new tokens, then resize via the model's own API (keeps a
        # tied MLM output head in sync -- see LGSEInitializer's docstring
        # for why hand-rolling this resize is unsafe)
        new_tokens = load_new_tokens(config.new_tokens_file)
        num_added = self.tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added}/{len(new_tokens)} new tokens to the tokenizer")
        self.model.resize_token_embeddings(len(self.tokenizer))

        embedding_layer = self.model.get_input_embeddings()
        embedding_dim = embedding_layer.embedding_dim
        old_vocab_size = embedding_layer.weight.shape[0] - num_added

        # 4) LGSE initialization: morpheme-average -> whole-token FastText
        # -> character n-grams
        morph_builder = MorphemeEmbeddingBuilder(
            fasttext_model=ft_model, segmenter=segmenter, embedding_dim=embedding_dim
        )
        char_encoder = CharNgramEncoder(
            n_min=config.ngram_min, n_max=config.ngram_max, dim=embedding_dim, device=str(self.device)
        )
        initializer = LGSEInitializer(
            embedding_layer=embedding_layer,
            morph_builder=morph_builder,
            char_encoder=char_encoder,
            device=str(self.device),
        )

        vocab = self.tokenizer.get_vocab()
        token_to_id = {tok: vocab[tok] for tok in new_tokens if tok in vocab}
        init_matrix = initializer.write_embeddings_for_new_tokens(token_to_id)

        # 5) regularizer anchored to the just-computed init vectors
        self.regularizer = LGSERegularizer(
            init_embeddings=init_matrix,
            token_ids=token_to_id,
            lambda_reg=config.reg_lambda,
            device=str(self.device),
        )

        # 6) freeze the whole model except the embedding matrix; gradients
        # for the *old* rows of that matrix get zeroed every step in
        # train_epoch() so only the new-token rows actually move --
        # "update only the new embeddings" from the paper, not "update
        # everything" (a previous version optimized model.parameters()
        # wholesale with no freezing at all).
        for p in self.model.parameters():
            p.requires_grad = False
        embedding_layer.weight.requires_grad = True
        self.embedding_layer = embedding_layer
        self.old_vocab_size = old_vocab_size

        self.optimizer = AdamW([embedding_layer.weight], lr=config.learning_rate)

        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=config.mlm_probability
        )
        self.dataloader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, collate_fn=self.collator
        )

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            mlm_loss = outputs.loss
            reg_loss = self.regularizer.loss(self.embedding_layer)
            loss = mlm_loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()

            if self.embedding_layer.weight.grad is not None:
                self.embedding_layer.weight.grad[: self.old_vocab_size] = 0

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[LGSELAPTrainer] avg loss this epoch: {avg_loss:.4f} "
              f"(mlm={mlm_loss.item():.4f} reg={reg_loss.item():.4f} on last batch)")
        return avg_loss

    def save(self, output_dir: str = None):
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Saved LGSE-specialized model to {output_dir}")
