from dataclasses import dataclass

@dataclass
class LGSEConfig:
    # Model + tokenizer
    model_name: str = "xlm-roberta-base"

    # Data files
    morph_lexicon_path: str = "data/morph_lexicon.txt"
    new_tokens_file: str = "data/new_tokens.txt"

    # FastText models (you choose which one to use)
    fasttext_amharic_path: str = "data/fasttext_amharic.bin"
    fasttext_tigrinya_path: str = "data/fasttext_tigrinya.bin"

    # Which language to specialize for this run: "am" or "ti"
    language: str = "am"

    # Training
    output_dir: str = "outputs/lgse_lap"
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_train_steps: int = 20000
    warmup_steps: int = 1000
    seed: int = 42
