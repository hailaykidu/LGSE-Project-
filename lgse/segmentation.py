from typing import Dict, List, Optional


class MorphologicalSegmenter:
    """
    Loads and provides morpheme segmentation for Amharic + Tigrinya words.

    The lexicon file mixes two sub-formats (confirmed against the actual
    data/morph_lexicon.txt, not assumed):

      Tigrinya rows: "[NUM] WORD [PREFIX] ROOT [SUFFIX]", space-separated,
      where PREFIX ends with '-' (e.g. "ኣይ-"), SUFFIX starts with '-'
      (e.g. "-ን"), a bare "-" means that slot is empty, and the leading row
      number is sometimes missing entirely for a handful of rows.

      Amharic rows (after a "$Amharic:" marker line): plain
      "WORD M1 M2 M3 ...", no hyphen markers at all.

    Rather than hardcoding fixed column positions (which breaks on the
    Tigrinya rows' variable field count -- 3 to 6 fields were observed),
    this drops a leading pure-integer row number if present, treats the
    next token as the word, and treats every remaining token as a
    morpheme after stripping any '-' markers, silently dropping bare "-"
    placeholders. This handles both sub-formats with one code path.
    """

    def __init__(self, lexicon: Dict[str, List[str]]):
        self.lexicon = lexicon

    @staticmethod
    def _parse_line(line: str) -> Optional[tuple]:
        tokens = line.split()
        if not tokens:
            return None
        if tokens[0].isdigit():
            tokens = tokens[1:]
        if not tokens:
            return None
        word = tokens[0]
        morphemes = [cleaned for t in tokens[1:] if (cleaned := t.strip("-"))]
        return word, morphemes

    @classmethod
    def from_file(cls, path: str) -> "MorphologicalSegmenter":
        lexicon: Dict[str, List[str]] = {}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Skip blank lines, comments, section markers (e.g.
                # "$Amharic:"), and the pipe-delimited header row.
                if not line or line.startswith("#") or line.startswith("$") or "|" in line:
                    continue

                parsed = cls._parse_line(line)
                if parsed is None:
                    continue
                word, morphemes = parsed
                if morphemes:
                    lexicon[word] = morphemes

        return cls(lexicon)

    def segment(self, token: str) -> List[str]:
        """
        Returns the morphemes for a token, or an empty list if no
        decomposition is known -- callers should treat an empty list as
        "fall back to character n-grams", not silently substitute the
        whole token as a fake single morpheme (the previous behavior,
        which made the character-n-gram fallback path unreachable).
        """
        if token in self.lexicon:
            return self.lexicon[token]

        lower = token.lower()
        if lower in self.lexicon:
            return self.lexicon[lower]

        return []
