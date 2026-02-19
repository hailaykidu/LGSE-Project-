from transformers import AutoTokenizer
from lgse import LGSEConfig
from lgse.token_selection import load_new_tokens

def main():
    config = LGSEConfig(
        base_model_name="xlm-roberta-base",
        new_tokens_file="data/new_tokens.txt",
        morph_lexicon_path="data/morph_lexicon.txt",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    new_tokens = load_new_tokens(config.new_tokens_file)
    added = tokenizer.add_tokens(new_tokens)

    print(f"Requested {len(new_tokens)} new tokens, tokenizer added {added}.")
    for tok in new_tokens[:20]:
        print(tok, "→", tokenizer.convert_tokens_to_ids(tok))

if __name__ == "__main__":
    main()
