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

    def __init__(self, lexicon: Dict[str, List[str]], analyzer=None):
        self.lexicon = lexicon
        # Optional morphological analyzer consulted only when the static
        # lexicon has no entry. See `with_hornmorpho`.
        self.analyzer = analyzer
        self._analyzer_cache: Dict[str, List[str]] = {}

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

    def with_hornmorpho(self) -> "MorphologicalSegmenter":
        """Attach HornMorpho (via amseg) as a fallback analyzer.

        The static lexicon is a fixed word list, so it decomposes only the
        words someone happened to enter: measured against the 198 Amharic
        tokens added by LAPT, it covers 45 (22.7%), leaving 77% of the new
        vocabulary to fall through to whole-token FastText -- which is
        approximately what the FOCUS baseline already does, so LGSE's
        morpheme path was barely exercised. HornMorpho *analyzes* rather
        than looks up, and covers 120/198 (60.6%) of the same list,
        subsuming every word the lexicon covers.

        Amharic only: amseg's analyzer is Amharic-specific. Tigrinya keeps
        the lexicon-only path, so its behavior is unchanged.

        Returns self unmodified if amseg is unavailable, so a missing
        optional dependency degrades to the previous behavior rather than
        failing a run.
        """
        try:
            from amseg.segmenter import AmharicSegmenter
        except Exception as exc:  # pragma: no cover - depends on environment
            print(f"[MorphologicalSegmenter] HornMorpho unavailable ({exc}); "
                  "using the static lexicon only")
            return self
        self.analyzer = AmharicSegmenter()
        return self

    def _analyze(self, token: str) -> List[str]:
        """Morphemes from the fallback analyzer, or [] if it cannot analyze.

        A word the analyzer declines to analyze comes back with status
        UNANALYZED, which must be treated as "no decomposition" rather
        than as a single whole-token morpheme -- otherwise every unanalyzed
        word would silently take the morpheme path with the token itself as
        its only morpheme, making the result identical to whole-token
        FastText while appearing to be a morpheme-derived embedding.
        """
        if token in self._analyzer_cache:
            return self._analyzer_cache[token]

        morphemes: List[str] = []
        try:
            from amseg.types import AnalysisStatus
            result = self.analyzer.segment_word(token)
            if (getattr(result, "status", None) == AnalysisStatus.ANALYZED
                    and result.analyses):
                morphemes = [m.text for m in result.analyses[0].morphemes
                             if m.text]
        except Exception:
            morphemes = []

        # A single morpheme equal to the token itself carries no
        # decomposition, so it is not a morpheme-path hit.
        if len(morphemes) == 1 and morphemes[0] == token:
            morphemes = []

        self._analyzer_cache[token] = morphemes
        return morphemes

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

        # The static lexicon wins where it has an entry, so attaching an
        # analyzer never changes a decomposition the lexicon already
        # supplied -- it only reaches words that previously fell through.
        if self.analyzer is not None:
            return self._analyze(token)

        return []
