from typing import List

def load_new_tokens(path: str) -> List[str]:
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok:
                tokens.append(tok)
    return tokens
