"""English text normalizer — rule-based using num2words.

Handles digits, years, decimals, ratios.
"""

import os
import re
from num2words import num2words


def _int_to_words(n: int) -> str:
    return num2words(n)


def _year_to_words(year: int) -> str:
    """Read 4-digit numbers as years (twenty sixteen, nineteen forty)."""
    if 1100 <= year <= 2999:
        return num2words(year, to='year')
    return num2words(year)


def normalize(text: str) -> str:
    """Normalize English text for TTS — verbalize all numeric expressions."""
    if not text:
        return text

    # Collapse spaces inside numbers like "10 000" → "10000"
    text = re.sub(r'(\d)\s+(\d{3})\b', r'\1\2', text)
    text = re.sub(r'(\d)\s+(\d{3})\b', r'\1\2', text)

    # Ratios "3:2" → "three to two"
    def ratio_repl(m):
        a, b = int(m.group(1)), int(m.group(2))
        return f"{_int_to_words(a)} to {_int_to_words(b)}"
    text = re.sub(r'\b(\d+):(\d+)\b(?!\d)', ratio_repl, text)

    # Decades "1960s" → "nineteen sixties"
    def decade_repl(m):
        year = int(m.group(1))
        # Pluralize properly: -ty → -ties
        words = num2words(year, to='year') if 1100 <= year <= 2999 else num2words(year)
        if words.endswith('y'):
            words = words[:-1] + 'ies'
        else:
            words = words + 's'
        return words
    text = re.sub(r'\b(\d{4})s\b', decade_repl, text)

    # Ordinals "1st", "2nd", "21st" (must run BEFORE unit-stripping)
    def ord_repl(m):
        return num2words(int(m.group(1)), to='ordinal')
    text = re.sub(r'\b(\d+)(st|nd|rd|th)\b', ord_repl, text)

    # Numbers stuck to units (35mm, 70km, m16, f1) — separate them
    text = re.sub(r'\b(\d+)([a-zA-Z]+)\b', lambda m: f"{_int_to_words(int(m.group(1)))} {m.group(2)}", text)
    text = re.sub(r'\b([a-zA-Z]+)(\d+)\b', lambda m: f"{m.group(1)} {_int_to_words(int(m.group(2)))}", text)

    # Decimals like "1.5" or "2.4"
    def decimal_repl(m):
        s = m.group(0)
        try:
            f = float(s)
            return num2words(f)
        except Exception:
            return s
    text = re.sub(r'\b\d+\.\d+\b', decimal_repl, text)

    # 4-digit years (in context of nearby year-like words, or just guess)
    def year_repl(m):
        n = int(m.group(0))
        return _year_to_words(n)
    text = re.sub(r'\b(1[0-9]{3}|20[0-9]{2})\b', year_repl, text)

    # All other plain integers
    def int_repl(m):
        n = int(m.group(0))
        return _int_to_words(n)
    text = re.sub(r'\b\d+\b', int_repl, text)

    # Cleanup: remove commas from "one thousand, nine hundred"
    text = text.replace(",", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if __name__ == "__main__":
    import json
    test_cases = [
        "however due to the slow communication channels styles in the west could lag behind by 25 to 30 year",
        "the aspect ratio of this format dividing by twelve to obtain the simplest whole-number ratio is therefore said to be 3:2",
        "through the night between 150 and 200 copies were made now known as dunlap broadsides",
        "congress began funding the obscenity initiative in fiscal 2005 and specified that the fbi must devote 10 agents",
        "in 1990 it was added to the list of world heritage sites",
        "in the 1960s a new wave began",
        "1.5 kilograms",
        "the 1st of may",
    ]

    print("=== TEST CASES ===")
    for raw in test_cases:
        result = normalize(raw)
        print(f"RAW:  {raw}")
        print(f"NORM: {result}")
        print()

    # Check FLEURS English coverage
    print("\n=== FLEURS English samples with digits ===")
    leftover = 0
    total = 0
    for path in [
        os.path.join(os.path.dirname(__file__), "eval_data", "test_lists", "fleurs_en_cross.jsonl"),
        os.path.join(os.path.dirname(__file__), "eval_data", "test_lists", "fleurs_en_regress.jsonl"),
    ]:
        for line in open(path):
            e = json.loads(line)
            t = e.get("text_raw", e["text"])
            if re.search(r'\d', t):
                total += 1
                normalized = normalize(t)
                if re.search(r'\d', normalized):
                    leftover += 1
                    if leftover <= 3:
                        print(f"\nLEFTOVER:")
                        print(f"  RAW:  {t}")
                        print(f"  NORM: {normalized}")
    print(f"\nLeftover: {leftover}/{total}, coverage {100*(1-leftover/total):.1f}%")
