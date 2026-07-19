import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lgse.segmentation import MorphologicalSegmenter

REAL_LEXICON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "morph_lexicon.txt")


def test_parses_tigrinya_prefix_root_suffix_row():
    # Matches an actual row in data/morph_lexicon.txt: "2 ኣይናቱን ኣይ- ናቱ -ን"
    seg = MorphologicalSegmenter({})
    result = seg._parse_line("2 ኣይናቱን ኣይ- ናቱ -ን")
    assert result == ("ኣይናቱን", ["ኣይ", "ናቱ", "ን"])


def test_parses_row_with_bare_hyphen_placeholder():
    # "1 ሰላማዊ - ሰላም -ኣዊ" -- the bare "-" (no prefix) must be dropped, not
    # kept as a literal "-" morpheme.
    seg = MorphologicalSegmenter({})
    result = seg._parse_line("1 ሰላማዊ - ሰላም -ኣዊ")
    assert result == ("ሰላማዊ", ["ሰላም", "ኣዊ"])


def test_parses_row_missing_leading_number():
    # A handful of rows in the real file lack the leading row number
    # entirely, e.g. "ኣይጀመረን ኣይ- ጀመረ -ኣን".
    seg = MorphologicalSegmenter({})
    result = seg._parse_line("ኣይጀመረን ኣይ- ጀመረ -ኣን")
    assert result == ("ኣይጀመረን", ["ኣይ", "ጀመረ", "ኣን"])


def test_parses_plain_amharic_row_no_hyphens():
    # Amharic-section rows are just "WORD M1 M2 M3", no hyphen markers.
    seg = MorphologicalSegmenter({})
    result = seg._parse_line("ተመለስኩ ተ መለስ ኩ")
    assert result == ("ተመለስኩ", ["ተ", "መለስ", "ኩ"])


def test_from_file_skips_comments_headers_and_markers():
    content = (
        "#ignore the number when you use it\n"
        "ተ/ቁ |ቃል(Word) |ቅድም ምእላድ(Prefix)|ሱር ቃል(Root word)|ድሕሪ ምእላድ(Suffix)\n"
        "1 ሰላማዊ - ሰላም -ኣዊ\n"
        "\n"
        "$Amharic:\n"
        "ተመለስኩ ተ መለስ ኩ\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(content)
        path = f.name
    try:
        seg = MorphologicalSegmenter.from_file(path)
        assert seg.lexicon == {"ሰላማዊ": ["ሰላም", "ኣዊ"], "ተመለስኩ": ["ተ", "መለስ", "ኩ"]}
    finally:
        os.unlink(path)


def test_segment_returns_empty_list_for_unknown_word():
    # Not [token] -- an empty list is what signals callers to fall back to
    # character n-grams. Returning [token] (the previous behavior) made
    # that fallback path unreachable, since a non-empty list always looked
    # like "found a segmentation" to callers.
    seg = MorphologicalSegmenter({"ሰላማዊ": ["ሰላም", "ኣዊ"]})
    assert seg.segment("ሰላማዊ") == ["ሰላም", "ኣዊ"]
    assert seg.segment("ኮምፒተር") == []


def test_from_file_against_real_lexicon():
    seg = MorphologicalSegmenter.from_file(REAL_LEXICON_PATH)
    # 226 non-comment/header/blank/marker lines in the real file, 210 of
    # them parse into a non-empty (word, morphemes) pair (verified by
    # direct inspection during the fix -- not an arbitrary number).
    assert len(seg.lexicon) == 210
    assert seg.segment("ሰላማዊ") == ["ሰላም", "ኣዊ"]
