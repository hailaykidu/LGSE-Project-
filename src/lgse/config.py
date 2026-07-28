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

    # FastText models are ~3 GB and are not committed. They are fetched by
    # data/scripts/download_fasttext.py, which records the resolved path,
    # dimension, vocab size and sha256 in data/fasttext_manifest.json.
    # Leaving these empty makes the manifest the single source of truth;
    # set them explicitly to override.
    fasttext_amharic_path: str = ""
    fasttext_tigrinya_path: str = ""
    fasttext_manifest: str = "data/fasttext_manifest.json"

    # Which language to specialize for this run: "am" or "ti"
    language: str = "am"

    # Character n-gram fallback (used when a token has no lexicon
    # segmentation and no FastText coverage at all)
    ngram_min: int = 3
    ngram_max: int = 5

    # LGSE regularization: penalizes drift of new-token embeddings away
    # from their initialized values during LAP (see LGSERegularizer)
    reg_lambda: float = 1.0

    # --- Table 2 system selection -------------------------------------
    # Which of the five compared systems this run is. The five differ only
    # in vocabulary expansion and new-token initialization; the backbone,
    # LAPT procedure and downstream fine-tuning are identical.
    system: str = "lgse_lapt"
    expand_vocab: bool = True
    initializer: str = "lgse"      # lgse | default | random | focus

    # The projection from FastText space to the model's embedding space is
    # the learned W of the LGSE method: trained jointly with the new
    # embeddings and saved with the checkpoint. It is deliberately not a
    # config option -- there is one projection, implemented in
    # src/lgse/projection.py.

    # Training
    output_dir: str = "outputs/lgse_lap"
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    mlm_probability: float = 0.15
    warmup_steps: int = 1000
    seed: int = 42
    device: str = "cuda"

    def _from_manifest(self, language: str) -> str:
        """Resolve a model path recorded by download_fasttext.py."""
        import json
        from pathlib import Path

        manifest = Path(self.fasttext_manifest)
        if not manifest.exists():
            raise SystemExit(
                f"{manifest} not found -- run\n"
                f"  python data/scripts/download_fasttext.py --language {language}\n"
                "to fetch the real FastText model. The repository ships no "
                "placeholder: a fake model would silently reduce LGSE to its "
                "character-n-gram fallback.")
        record = json.load(open(manifest, encoding="utf-8")).get(language)
        if not record:
            raise SystemExit(f"{manifest} has no entry for {language!r}")

        # Fail here rather than after loading a multi-GB model: the manifest
        # already records the dimension, so an incompatible model can be
        # rejected before any expensive work happens. The authoritative
        # check still runs in build_projection; this is an early, cheaper
        # copy of it.
        dim = record.get("dimension")
        if dim is not None:
            from .projection import check_dimensions
            check_dimensions(dim, self.embedding_dim,
                             source=f"{language} FastText, {record['path']}")
        return record["path"]

    @property
    def embedding_dim(self) -> int:
        """The target model's embedding width.

        FastText must match this exactly, because LGSE's W is square
        (paper Sec 4.1). Declared here so the requirement can be checked
        before a model is loaded; `xlm-roberta-base` is the paper's model.
        """
        known = {"xlm-roberta-base": 768, "xlm-roberta-large": 1024}
        if self.model_name in known:
            return known[self.model_name]
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(self.model_name).hidden_size

    @property
    def fasttext_path(self) -> str:
        if self.language == "am":
            return self.fasttext_amharic_path or self._from_manifest("amharic")
        elif self.language == "ti":
            return self.fasttext_tigrinya_path or self._from_manifest("tigrinya")
        raise ValueError(f"Unknown language: {self.language!r} (expected 'am' or 'ti')")
