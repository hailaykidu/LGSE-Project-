from typing import List


def load_new_tokens(path: str) -> List[str]:
    """The single canonical definition -- a previous version of this
    project had a second, duplicate copy of this function inline in
    lap_trainer.py that additionally deduped/sorted; that behavior is kept
    here since deterministic ordering matters for reproducible runs.
    """
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok:
                tokens.append(tok)
    return sorted(set(tokens))
