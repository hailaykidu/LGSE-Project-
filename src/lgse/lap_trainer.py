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
from .projection import build_projection
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
    Language-Adaptive Pretraining (LAPT) with LGSE-initialized embeddings.

    This implementation follows the LGSE framework described in the
    accompanying paper, integrating morphology-aware decomposition,
    FastText-based representations, and character n-gram fallback for
    initializing new vocabulary embeddings. During LAPT, embedding
    regularization preserves the initialized semantic structure while
    adapting the new representations to target language data.
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

        # A system that does not expand the vocabulary (the XLM-R baseline)
        # has no new tokens to initialize and no LAPT stage: it is the
        # unmodified backbone, and Table 2 reports it as such.
        self.system = getattr(config, "system", "lgse_lapt")
        self.expand_vocab = getattr(config, "expand_vocab", True)
        self.initializer_kind = getattr(config, "initializer", "lgse")

        if not self.expand_vocab:
            self.regularizer = None
            self.embedding_layer = self.model.get_input_embeddings()
            self.old_vocab_size = self.embedding_layer.weight.shape[0]
            self.optimizer = None
            self.dataloader = None
            print(f"[LGSELAPTrainer] system={self.system}: no vocabulary "
                  f"expansion, no LAPT")
            return

        # 2) lexicon + FastText. Only the LGSE and FOCUS paths need them:
        # the random and default initializers use no external signal.
        needs_fasttext = self.initializer_kind in ("lgse", "focus")
        segmenter = MorphologicalSegmenter.from_file(config.morph_lexicon_path)
        print(f"Loaded morphological lexicon: {len(segmenter.lexicon)} words")
        ft_model = fasttext.load_model(config.fasttext_path) \
            if needs_fasttext else None

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

        # 4) the learned projection W, trained jointly with the new
        # embeddings (see src/lgse/projection.py). It is only needed on the
        # paths that consume FastText vectors.
        self.projection = None
        if ft_model is not None:
            self.projection = build_projection(
                source_dim=ft_model.get_dimension(),
                target_dim=embedding_dim,
                seed=config.seed,
            )
            if self.projection is not None:
                self.projection.to(self.device)
                n_params = sum(p.numel() for p in self.projection.parameters()
                               if p.requires_grad)
                print(f"[LGSELAPTrainer] learned projection W: "
                      f"{ft_model.get_dimension()} -> {embedding_dim} "
                      f"({n_params} trainable parameters)")

        # 5) initialization: LGSE, or one of the Table 2 baselines
        vocab = self.tokenizer.get_vocab()
        token_to_id = {tok: vocab[tok] for tok in new_tokens if tok in vocab}

        if self.initializer_kind == "lgse":
            morph_builder = MorphemeEmbeddingBuilder(
                fasttext_model=ft_model, segmenter=segmenter,
                embedding_dim=embedding_dim, seed=config.seed,
                projection=self.projection,
            )
            char_encoder = CharNgramEncoder(
                n_min=config.ngram_min, n_max=config.ngram_max,
                dim=embedding_dim, device=str(self.device),
            )
            initializer = LGSEInitializer(
                embedding_layer=embedding_layer,
                morph_builder=morph_builder,
                char_encoder=char_encoder,
                device=str(self.device),
            )
        else:
            from baselines import build_initializer
            kwargs = {"embedding_layer": embedding_layer,
                      "device": str(self.device)}
            if self.initializer_kind == "random":
                kwargs.update(seed=config.seed, old_vocab_size=old_vocab_size)
            elif self.initializer_kind == "focus":
                kwargs.update(aux_vectors=None, aux_index=None,
                              old_vocab_size=old_vocab_size)
            initializer = build_initializer(self.initializer_kind, **kwargs)

        init_matrix = initializer.write_embeddings_for_new_tokens(token_to_id)

        # 6) regularizer anchored to the lexically grounded targets.
        #
        # For LGSE the anchor is recomputed through W on every step rather
        # than frozen at its initial value. This is what puts W on the
        # training graph at all: the initializer writes its output into the
        # embedding via `.data`, which severs the graph, so a detached
        # anchor would leave W with no gradient source anywhere in LAPT --
        # trainable on paper, frozen in fact.
        #
        # The baselines have no projection, so their anchor stays a fixed
        # tensor and their behaviour is unchanged.
        if self.projection is not None and self.initializer_kind == "lgse":
            tokens = list(token_to_id.keys())

            def live_anchor(_tokens=tokens, _init=initializer):
                return torch.stack(
                    [_init.init_token_embedding(t) for t in _tokens], dim=0)

            anchor = live_anchor
        else:
            anchor = init_matrix

        self.regularizer = LGSERegularizer(
            init_embeddings=anchor,
            token_ids=token_to_id,
            lambda_reg=config.reg_lambda,
            device=str(self.device),
        )
        print(f"[LGSELAPTrainer] regularizer anchor: "
              f"{'live through W' if self.regularizer.anchor_is_live else 'fixed'}")

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

        # The learned projection is part of the optimization problem: if its
        # parameters are not handed to the optimizer, W never moves and
        # "learned" silently degrades to "randomly initialized and frozen".
        trainable = [embedding_layer.weight]
        if self.projection is not None and self.projection.is_learned:
            trainable += list(self.projection.parameters())
        self.optimizer = AdamW(trainable, lr=config.learning_rate)

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

    PROJECTION_FILE = "projection.pt"

    def save(self, output_dir: str = None):
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # W is a trained parameter of the method, not a derived artifact: it
        # cannot be recomputed from the seed once it has been optimized.
        # save_pretrained() only covers the model, so without this the
        # trained mapping is lost and a restored run silently falls back to
        # a fresh initialization.
        if self.projection is not None:
            path = os.path.join(output_dir, self.PROJECTION_FILE)
            torch.save({
                "state_dict": self.projection.state_dict(),
                "source_dim": self.projection.source_dim,
                "target_dim": self.projection.target_dim,
            }, path)
            print(f"Saved learned projection W to {path}")

        print(f"Saved LGSE-specialized model to {output_dir}")

    def load_projection(self, output_dir: str = None):
        """Restore a trained W saved by `save()`.

        Raises if the checkpoint's shape disagrees with the projection this
        run built -- a silent shape mismatch would mean loading someone
        else's mapping.
        """
        output_dir = output_dir or self.config.output_dir
        path = os.path.join(output_dir, self.PROJECTION_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no saved projection at {path}; the checkpoint was written "
                "without one, so the trained W is unavailable.")
        # weights_only=True: the checkpoint holds tensors and two ints, so
        # there is no reason to allow arbitrary pickle execution here.
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if self.projection is None:
            raise RuntimeError(
                "this run built no projection (FastText and the embedding "
                "space share a width), so there is nothing to restore into.")
        if (ckpt["source_dim"], ckpt["target_dim"]) != (
                self.projection.source_dim, self.projection.target_dim):
            raise ValueError(
                f"projection shape mismatch: checkpoint is "
                f"{ckpt['source_dim']}->{ckpt['target_dim']}, this run "
                f"expects {self.projection.source_dim}->"
                f"{self.projection.target_dim}")
        self.projection.load_state_dict(ckpt["state_dict"])
        self.projection.to(self.device)
        print(f"Restored learned projection W from {path}")
        return self.projection
