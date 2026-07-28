from typing import List

def expand_vocab_with_tokens(tokenizer, tokens: List[str]) -> int:
    """
    Simple wrapper around tokenizer.add_tokens for compatibility.
    Returns number of tokens added.
    """
    return tokenizer.add_tokens(tokens)
