from dataclasses import dataclass

# Sentinel for a parameter the caller must set explicitly. A dataclass field
# cannot be required once earlier fields carry defaults, so the requirement
# is enforced in __post_init__ instead.
REQUIRED = float("nan")


class MissingRequiredParameter(ValueError):
    """Raised when a mandatory configuration value was not supplied."""


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

    # LGSE regularization strength: lambda in the paper's
    # L_reg = lambda * ||e_new - mu||^2 (Sec 4.2).
    #
    # MANDATORY -- there is no default. The paper introduces lambda but
    # never assigns it, and Table 1 does not list it, so any value is the
    # experimenter's choice rather than the paper's. A silent default would
    # bury that choice in a dataclass field and make every result look as if
    # it followed a published setting. Callers must state it explicitly, and
    # the value is recorded per run.
    reg_lambda: float = REQUIRED

    # --- Table 2 system selection -------------------------------------
    # Which of the five compared systems this run is. The five differ only
    # in vocabulary expansion and new-token initialization; the backbone,
    # LAPT procedure and downstream fine-tuning are identical.
    system: str = "lgse_lapt"
    expand_vocab: bool = True
    initializer: str = "lgse"      # lgse | default | random | focus

    # Alignment matrix W (paper Sec 4.1). MANDATORY for any system that uses
    # FastText: point this at a .pt/.npy file holding a d x d matrix.
    #
    # There is no default -- not even the identity. The paper introduces W
    # but never says how it is obtained, so any matrix chosen here would be
    # this implementation's decision rather than the authors'. Runs without
    # one fail; see build_projection() for the full explanation.
    #
    # W is frozen regardless: no objective in the paper trains it.
    # See src/lgse/projection.py and DEVIATIONS.md section 1a.
    alignment_matrix_path: str = ""

    # Training
    output_dir: str = "outputs/lgse_lap"
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    mlm_probability: float = 0.15
    warmup_steps: int = 1000
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        # NaN is the sentinel, and NaN != NaN, so this catches "not set"
        # without rejecting any real value the experimenter might choose.
        if self.reg_lambda != self.reg_lambda:
            raise MissingRequiredParameter(
                "reg_lambda is mandatory and was not supplied.\n"
                "\n"
                "It is lambda in the paper's regularization term\n"
                "    L_reg = lambda * ||e_new - mu||^2   (Sec 4.2)\n"
                "\n"
                "The paper introduces lambda but never states its value, and "
                "Table 1 does not list it, so there is no published setting "
                "to default to. Choosing one silently would present an "
                "experimenter's choice as the paper's.\n"
                "\n"
                "Set it explicitly, e.g. LGSEConfig(reg_lambda=1.0, ...) or "
                "`lgse.reg_lambda` in the run config. The value is recorded "
                "with every result. See DEVIATIONS.md section 8.")
        if self.reg_lambda < 0:
            raise ValueError(
                f"reg_lambda must be non-negative, got {self.reg_lambda}")

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
