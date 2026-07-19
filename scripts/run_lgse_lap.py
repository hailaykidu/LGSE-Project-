import sys
from pathlib import Path

from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lgse import LGSEConfig, LGSELAPTrainer


class SimpleMLMDataset(Dataset):
    """
    Tokenizes raw text lines for masked-language-model training. Returns
    unpadded, unmasked token ids -- LGSELAPTrainer's DataCollatorForLanguageModeling
    does the actual random masking and padding per batch. A previous version
    of this dataset padded to a fixed length and set labels = input_ids
    directly in __getitem__, which meant no masking was ever applied at
    all: the "MLM" loss was really just reconstructing the unmasked input.

    Replace `texts` with a real corpus for an actual experiment -- e.g.
    the already-collected, cleaned Amharic corpus at
    ../../GeezTokenizer/02_cleaning/corpus_clean/amharic.txt (12.19M lines).
    The two example sentences below are only enough to smoke-test that the
    pipeline runs end to end, not to produce a meaningful trained model.
    """

    def __init__(self, texts, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        return {"input_ids": enc["input_ids"]}


def load_corpus(path: str = None):
    if path is None:
        print("[run_lgse_lap] No --corpus_file given; using 2 placeholder "
              "sentences for a smoke test only. Pass a real corpus for an "
              "actual experiment.")
        return [
            "Example sentence in the target language.",
            "Replace this with real Amharic/Tigrinya data.",
        ]
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", default=None,
                         help="Path to a plain-text corpus, one sentence per line. "
                              "Defaults to a 2-sentence smoke-test placeholder.")
    parser.add_argument("--language", default="am", choices=["am", "ti"])
    parser.add_argument("--num_train_epochs", type=int, default=None)
    args = parser.parse_args()

    config = LGSEConfig(language=args.language)
    if args.num_train_epochs is not None:
        config.num_train_epochs = args.num_train_epochs

    texts = load_corpus(args.corpus_file)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = SimpleMLMDataset(texts, tokenizer)

    trainer = LGSELAPTrainer(config, dataset)

    for epoch in range(config.num_train_epochs):
        print(f"Epoch {epoch + 1}/{config.num_train_epochs}")
        trainer.train_epoch()

    trainer.save()


if __name__ == "__main__":
    main()
