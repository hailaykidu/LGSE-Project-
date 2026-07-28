from dataclasses import dataclass


@dataclass
class LGSEConfig:
    """
    Single canonical config -- a previous version of this project had two
    conflicting LGSEConfig dataclasses (this file and lgse/__init__.py)
    with different field names, so whichever one a given script imported
    silently expected different attributes than what lap_trainer.py
    actually read. lgse/__init__.py now re-exports this definition instead
    of defining its own.
    """

    # Model + tokenizer
    model_name: str = "xlm-roberta-base"

    # Data files
    morph_lexicon_path: str = "data/morph_lexicon.txt"
    new_tokens_file: str = "data/new_tokens.txt"

    # FastText models -- filenames matched against what's actually on disk
    # (data/fasttext_Amharic.bin, data/fasttext_Tigriyna.bin), not assumed;
    # the previous defaults here were lowercase and didn't match either
    # actual file, so loading with the defaults raised FileNotFoundError.
    fasttext_amharic_path: str = "data/fasttext_Amharic.bin"
    fasttext_tigrinya_path: str = "data/fasttext_Tigriyna.bin"

    # Which language to specialize for this run: "am" or "ti"
    language: str = "am"

    # Character n-gram fallback (used when a token has no lexicon
    # segmentation and no FastText coverage at all)
    ngram_min: int = 3
    ngram_max: int = 5

    # LGSE regularization: penalizes drift of new-token embeddings away
    # from their initialized values during LAP (see LGSERegularizer)
    reg_lambda: float = 1.0

    # Training
    output_dir: str = "outputs/lgse_lap"
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    mlm_probability: float = 0.15
    warmup_steps: int = 1000
    seed: int = 42
    device: str = "cuda"

    @property
    def fasttext_path(self) -> str:
        if self.language == "am":
            return self.fasttext_amharic_path
        elif self.language == "ti":
            return self.fasttext_tigrinya_path
        raise ValueError(f"Unknown language: {self.language!r} (expected 'am' or 'ti')")
