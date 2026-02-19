import os
import random
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AdamW
import fasttext

from config import LGSEConfig
from morph_lexicon import MorphologicalSegmenter
from morpheme_embeddings import MorphemeEmbeddingBuilder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_new_tokens(path: str):
    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok:
                tokens.append(tok)
    return sorted(set(tokens))


def load_fasttext_for_language(config: LGSEConfig):
    if config.language == "am":
        path = config.fasttext_amharic_path
    elif config.language == "ti":
        path = config.fasttext_tigrinya_path
    else:
        raise ValueError(f"Unknown language: {config.language}")
    print(f"Loading FastText model from {path}")
    return fasttext.load_model(path)


def main():
    config = LGSEConfig()
    set_seed(config.seed)

    # 1) load base model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)

    # 2) load lexicon
    segmenter = MorphologicalSegmenter.from_file(config.morph_lexicon_path)

    # 3) load FastText
    ft_model = load_fasttext_for_language(config)

    # 4) load new tokens
    new_tokens = load_new_tokens(config.new_tokens_file)
    print(f"Loaded {len(new_tokens)} new tokens")

    # 5) add new tokens to tokenizer
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added} tokens to tokenizer")

    # 6) resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    embedding_layer = model.get_input_embeddings()
    embedding_dim = embedding_layer.embedding_dim

    # 7) build LGSE embeddings for new tokens
    builder = MorphemeEmbeddingBuilder(
        fasttext_model=ft_model,
        segmenter=segmenter,
        embedding_dim=embedding_dim,
    )

    token_to_id = tokenizer.get_vocab()
    with torch.no_grad():
        for tok in new_tokens:
            if tok not in token_to_id:
                continue
            idx = token_to_id[tok]
            vec = builder.build_embedding_for_token(tok)
            vec_t = torch.tensor(vec, dtype=embedding_layer.weight.dtype)
            embedding_layer.weight[idx] = vec_t

    # 8) LAP training loop (simplified: you plug your corpus here)
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # TODO: replace this with your real Amharic/Tigrinya corpus loader
    # Here we just show the structure.
    for step in range(config.num_train_steps):
        # dummy loss to illustrate
        loss = torch.zeros(1, requires_grad=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}/{config.num_train_steps}")

    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Saved LGSE-specialized model to {config.output_dir}")


if __name__ == "__main__":
    main()
