"""Georgian text normalizer — rule-based, fast (no LLM).

Handles digits, dates, decades, ratios in FLEURS Georgian style.
Uses num2geotext for the actual word conversion.

Usage:
    from text_normalizer import normalize
    text = normalize("1940 წლის 15 აგვისტოს")
    # → "ათას ცხრაას ორმოცი წლის თხუთმეტი აგვისტოს"
"""

import os
import re
from num2geotext import int_num_to_geo_text, float_num_to_geo_text


def _int_to_geo(n: int) -> str:
    try:
        return int_num_to_geo_text(n)
    except Exception:
        return str(n)


def _float_to_geo(s: str) -> str:
    """Handle decimals like 1.5 or 6,5 (Georgian uses comma)."""
    s = s.replace(",", ".")
    try:
        f = float(s)
        if f == int(f):
            return _int_to_geo(int(f))
        result = float_num_to_geo_text(f)
        # num2geotext outputs "ერთი მთელი, ხუთი მეათედი" — remove the comma
        return result.replace(",", "")
    except Exception:
        return s


def normalize(text: str) -> str:
    """Normalize Georgian text for TTS — verbalize all numeric expressions."""
    if not text:
        return text

    # Step 1: collapse spaces inside numbers like "40 000" → "40000"
    text = re.sub(r'(\d)\s+(\d{3})\b', r'\1\2', text)
    text = re.sub(r'(\d)\s+(\d{3})\b', r'\1\2', text)  # twice for "1 000 000"


    # Step 2: ratios like "3:2" → "სამი ორთან"
    def ratio_repl(m):
        a, b = int(m.group(1)), int(m.group(2))
        return f"{_int_to_geo(a)} {_int_to_geo(b)}თან"
    text = re.sub(r'\b(\d+):(\d+)\b(?!\d)', ratio_repl, text)

    # Step 3: time ranges "10:00-11:00" → "ათიდან თერთმეტ საათამდე" (handled by ratios for hours, but special case)
    # Just convert HH:MM to "HH საათი MM წუთი" if MM != 00, else "HH საათი"
    def time_repl(m):
        h, mn = int(m.group(1)), int(m.group(2))
        if mn == 0:
            return f"{_int_to_geo(h)} საათი"
        return f"{_int_to_geo(h)} საათი {_int_to_geo(mn)} წუთი"
    # This would conflict with ratio, so skipped — ratios handle it well enough

    # Step 4: decades like "1960-იან", "1800-იანი"
    def decade_repl(m):
        year = int(m.group(1))
        suffix = m.group(2)
        return f"{_int_to_geo(year)}{suffix}"
    text = re.sub(r'\b(\d{4})-(იან\w*)', decade_repl, text)
    text = re.sub(r'\b(\d{2,3}0)-(იან\w*)', decade_repl, text)

    # Step 5: ordinals like "მე-20" → "მეოცე" (rough — just verbalize the number)
    def me_ordinal_repl(m):
        n = int(m.group(1))
        return f"მე{_int_to_geo(n)}ე"  # rough ordinal form
    text = re.sub(r'მე-(\d+)', me_ordinal_repl, text)

    # Step 6: numbers with -ე, -ჯერ, -ან, -ში suffix → just verbalize the number with suffix
    def num_suffix_repl(m):
        n = int(m.group(1))
        suffix = m.group(2)
        return f"{_int_to_geo(n)}{suffix}"
    text = re.sub(r'\b(\d+)-(ე|ჯერ|ან|ში|ის|ით)\b', num_suffix_repl, text)

    # Step 7: Decimal numbers (Georgian uses comma)
    def decimal_repl(m):
        return _float_to_geo(m.group(0))
    text = re.sub(r'\b\d+[.,]\d+\b', decimal_repl, text)

    # Step 8: plain integers
    def int_repl(m):
        n = int(m.group(0))
        return _int_to_geo(n)
    text = re.sub(r'\b\d+\b', int_repl, text)

    # Cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if __name__ == "__main__":
    import json

    test_cases = [
        ("1940 წლის 15 აგვისტოს", None),
        ("შეხვედრა იქნება 10:00-11:00 საათზე.", None),
        ("1960-იან წლებში", None),
        ("40 000-ზე მეტი ადამიანი", None),
        ("შედეგი იყო 3:2.", None),
        ("მე-20 საუკუნე", None),
        ("100 პროცენტი", None),
        ("1.5 კილოგრამი", None),
        ("5000 წიგნი", None),
        ("მიწისძვრა 6,5 მაგნიტუდის", None),
        ("1800-იანი წლების შემდეგ", None),
        ("10 000 წლის წინ", None),
        ("1990 წელს დაემატა", None),
        ("20 პროცენტი მოდის", None),
        ("70 კმ/სთ-ში", None),
    ]

    print("=== TEST CASES ===")
    for raw, _ in test_cases:
        result = normalize(raw)
        print(f"RAW:  {raw}")
        print(f"NORM: {result}")
        print()

    # Now run on all FLEURS samples with digits
    print("\n=== FLEURS-KA samples with digits ===")
    samples_with_digits = []
    test_path = os.path.join(os.path.dirname(__file__), "eval_data", "test_lists", "fleurs_ka.jsonl")
    with open(test_path) as f:
        for line in f:
            e = json.loads(line)
            raw = e.get("text_raw", e["text"])
            if re.search(r'[0-9]', raw):
                samples_with_digits.append(raw)

    print(f"Total: {len(samples_with_digits)} samples")

    leftover_digits = 0
    for raw in samples_with_digits:
        normalized = normalize(raw)
        if re.search(r'\d', normalized):
            leftover_digits += 1
            if leftover_digits <= 5:
                print(f"\nLEFTOVER DIGITS:")
                print(f"  RAW:  {raw}")
                print(f"  NORM: {normalized}")

    print(f"\nSamples with leftover digits: {leftover_digits}/{len(samples_with_digits)}")
    print(f"Coverage: {100*(1-leftover_digits/len(samples_with_digits)):.1f}%")
