import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lgse import LGSEConfig
from lgse.token_selection import load_new_tokens


def main():
    config = LGSEConfig(
        model_name="xlm-roberta-base",
        new_tokens_file="data/new_tokens.txt",
        morph_lexicon_path="data/morph_lexicon.txt",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    new_tokens = load_new_tokens(config.new_tokens_file)
    added = tokenizer.add_tokens(new_tokens)

    print(f"Requested {len(new_tokens)} new tokens, tokenizer added {added}.")
    for tok in new_tokens[:20]:
        print(tok, "→", tokenizer.convert_tokens_to_ids(tok))


if __name__ == "__main__":
    main()
