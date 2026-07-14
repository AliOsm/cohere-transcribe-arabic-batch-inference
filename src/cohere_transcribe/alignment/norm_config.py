"""MMS text-normalization patterns used by Arabic and English alignment."""

from __future__ import annotations

import re
from pathlib import Path

_PUNCTUATION_PREFIX = (
    r"\.\?,:!{}"
    r"\u201c\u201d\u00ab\u00bb\u2039\u203a\u201e\u201a\u201f\u201b\u2019\u2018"
    r"\u003e\u003c\u00BF\u3002;"
    r"\u055A\u055B\u055C\u055D\u055E\u055F\u0589"
    r"\u00A1\u060C\u3001।\"\u061B\u061F"
    r"\u3002\uFF0C\uFF01\uFF1F\uFF1B\uFF1A\uFF08\uFF09"
    r"\u300C-\u300F\uFF41-\uFF44\u3008-\u300B"
    r"\uFE4F\u22EF\u3001\u2027\uFF0F\uFF5E\u2500\uFF3F"
)

with (
    Path(__file__)
    .with_name("punctuations.lst")
    .open(encoding="utf-8-sig") as punctuation_file
):
    _MMS_PUNCTUATION = "".join(
        re.escape(line.split("\t", 1)[0]) for line in punctuation_file
    )

PUNCTUATION_SET = _PUNCTUATION_PREFIX + _MMS_PUNCTUATION
DELETION_SET = r"\u200E\u200C\u0656-\u0657\u200B\u064B-\u0652\u202C\u200F\u202A"
DIGIT_SET = (
    r"\u0030-\u0039\u09e6-\u09ef\u17e0-\u17e9\u0966-\u096f"
    r"\u0b66-\u0b6f\u06f0-\u06f9\ua900-\ua909\uff10-\uff19"
    r"\u0d66-\u0d6f\u1040-\u1049\u2170-\u2179\u206f"
)
MAPPINGS = {
    "&lt;": "",
    "&gt;": "",
    "&nbsp": "",
    r"(\S+)[\u201b\u2019\u2018](\S+)": r"\1'\2",
    "ٱ": "ا",  # noqa: RUF001 - Arabic normalization mapping
    "ٰ": "ا",  # noqa: RUF001 - Arabic normalization mapping
    "ۥ": "و",
    "ۦ": "ي",
    "ـ": "",
    "ٓ": "",
    "ٔ": "ء",
    "ٕ": "ء",
}
