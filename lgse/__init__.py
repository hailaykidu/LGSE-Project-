from dataclasses import dataclass
from typing import Optional

@dataclass
class LGSEConfig:
    base_model_name: str
    new_tokens_file: str
    morph_lexicon_path: str
    fasttext_path: Optional[str] = None
    ngram_min: int = 3
    ngram_max: int = 5
    reg_lambda: float = 1.0
    lr: float = 5e-5
    batch_size: int = 16
    num_epochs: int = 3
    device: str = "cuda"
