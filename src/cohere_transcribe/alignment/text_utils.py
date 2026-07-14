"""Arabic and English MMS text normalization and timestamp postprocessing."""

from __future__ import annotations

import re
import unicodedata
from itertools import pairwise

from uroman import Uroman

from .norm_config import DELETION_SET, DIGIT_SET, MAPPINGS, PUNCTUATION_SET

uroman_instance = Uroman()


def text_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\([^)]*\d[^)]*\)", " ", text)
    for old, new in MAPPINGS.items():
        text = re.sub(old, new, text)

    normalized = re.sub(f"[{PUNCTUATION_SET}]", " ", text)
    normalized = re.sub(f"[{DELETION_SET}]", "", normalized)
    digits = f"[{DIGIT_SET}]+"
    normalized = re.sub(
        rf"^{digits}(?=\s)|(?<=\s){digits}(?=\s)|(?<=\s){digits}$",
        " ",
        normalized,
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _uroman_token(text: str, language: str) -> str:
    romanized = uroman_instance.romanize_string(text, lcode=language)
    romanized = " ".join(romanized.strip()).lower()
    romanized = re.sub(r"[^a-z' ]", " ", romanized)
    return re.sub(r" +", " ", romanized).strip()


def preprocess_text(text: str, language: str) -> tuple[list[str], list[str]]:
    words = text.split()
    tokens = [_uroman_token(text_normalize(word), language) for word in words]
    tokens_starred: list[str] = []
    text_starred: list[str] = []
    for token, word in zip(tokens, words, strict=True):
        tokens_starred.extend(("<star>", token))
        text_starred.extend(("<star>", word))
    return tokens_starred, text_starred


def postprocess_results(
    text_starred: list[str],
    spans: list,
    stride: float,
) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for index, text in enumerate(text_starred):
        if text == "<star>":
            continue
        span = spans[index]
        results.append(
            {
                "start": span[0].start * stride / 1000,
                "end": (span[-1].end + 1) * stride / 1000,
                "text": text,
            }
        )

    for current, following in pairwise(results):
        if following["start"] < current["end"]:
            following["start"] = current["end"]
    return results
