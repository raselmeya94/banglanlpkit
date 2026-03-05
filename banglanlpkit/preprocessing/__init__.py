"""
banglanlp.preprocessing
========================
Text preprocessing utilities specifically designed for Bangla (Bengali) text.
Handles Unicode normalization, cleaning, stopword removal, and pipeline building.
"""

import re
import unicodedata
from typing import List, Optional, Union, Callable
from dataclasses import dataclass, field


# ── Unicode ranges ──────────────────────────────────────────────────────────
BANGLA_UNICODE_RANGE = (0x0980, 0x09FF)
BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
ARABIC_TO_BANGLA = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
BANGLA_TO_ARABIC = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

BANGLA_STOPWORDS = {
    "এবং", "বা", "কিন্তু", "তবে", "যদি", "তাহলে", "কারণ", "যেহেতু",
    "আর", "না", "নয়", "হয়", "হবে", "হয়েছে", "করা", "করে", "করেছে",
    "আমি", "তুমি", "সে", "আমরা", "তোমরা", "তারা", "এই", "ওই", "সেই",
    "যে", "যা", "যার", "এটা", "ওটা", "সেটা", "এখানে", "সেখানে",
    "এখন", "তখন", "সব", "সবাই", "কেউ", "কিছু", "অনেক", "খুব",
    "আছে", "ছিল", "থাকে", "হলো", "গেল", "এলো", "দিল", "নিল",
    "একটি", "একটা", "একজন", "প্রতি", "থেকে", "দিয়ে", "নিয়ে",
    "পর", "আগে", "পরে", "সাথে", "কাছে", "মধ্যে", "জন্য", "উপর",
}


class BanglaTextCleaner:
    """
    Clean Bangla text by removing noise, URLs, emails, HTML tags, and
    optionally English characters.

    Parameters
    ----------
    remove_urls : bool
        Strip HTTP/HTTPS URLs from text (default True).
    remove_emails : bool
        Strip email addresses (default True).
    remove_html : bool
        Strip HTML tags (default True).
    remove_english : bool
        Remove ASCII/English characters (default False).
    remove_numbers : bool
        Remove numeric characters including Bangla digits (default False).
    remove_special_chars : bool
        Remove punctuation and special symbols (default False).

    Examples
    --------
    >>> cleaner = BanglaTextCleaner(remove_urls=True, remove_html=True)
    >>> cleaner.clean("আমাদের ওয়েবসাইট https://example.com দেখুন।")
    'আমাদের ওয়েবসাইট  দেখুন।'
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_html: bool = True,
        remove_english: bool = False,
        remove_numbers: bool = False,
        remove_special_chars: bool = False,
    ):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.remove_english = remove_english
        self.remove_numbers = remove_numbers
        self.remove_special_chars = remove_special_chars

    def clean(self, text: str) -> str:
        """Clean the input Bangla text."""
        if self.remove_html:
            text = re.sub(r"<[^>]+>", " ", text)
        if self.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        if self.remove_emails:
            text = re.sub(r"\S+@\S+\.\S+", " ", text)
        if self.remove_english:
            text = re.sub(r"[a-zA-Z]", " ", text)
        if self.remove_numbers:
            text = re.sub(r"[0-9]", " ", text)
            text = re.sub(f"[{BANGLA_DIGITS}]", " ", text)
        if self.remove_special_chars:
            text = re.sub(r"[^\u0980-\u09FF\s]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def __call__(self, text: str) -> str:
        return self.clean(text)

    def __repr__(self):
        return (
            f"BanglaTextCleaner("
            f"remove_urls={self.remove_urls}, "
            f"remove_html={self.remove_html}, "
            f"remove_english={self.remove_english})"
        )


class BanglaNormalizer:
    """
    Normalize Bangla Unicode text.

    Handles:
    - Hasanta (্) normalization
    - Zero-width non-joiner / joiner cleanup
    - Visarga / Anusvara variants
    - Digit conversion (Arabic ↔ Bangla)
    - Punctuation normalization

    Parameters
    ----------
    normalize_digits : bool
        Convert Arabic numerals to Bangla (default False).
    normalize_punctuation : bool
        Normalize punctuation marks (default True).
    remove_zwj : bool
        Strip zero-width joiners/non-joiners (default True).

    Examples
    --------
    >>> normalizer = BanglaNormalizer(normalize_digits=True)
    >>> normalizer.normalize("আমার বয়স 25 বছর")
    'আমার বয়স ২৫ বছর'
    """

    # Common character substitution map for Bangla
    CHAR_MAP = {
        "\u09BC": "",      # nukta – strip for standard text
        "\u200C": "",      # ZWNJ
        "\u200D": "",      # ZWJ
        "\u00A0": " ",     # non-breaking space → regular space
        "\u2013": "-",     # en-dash
        "\u2014": "-",     # em-dash
        "\u201C": '"',     # left double quote
        "\u201D": '"',     # right double quote
        "\u2018": "'",     # left single quote
        "\u2019": "'",     # right single quote
    }

    def __init__(
        self,
        normalize_digits: bool = False,
        normalize_punctuation: bool = True,
        remove_zwj: bool = True,
    ):
        self.normalize_digits = normalize_digits
        self.normalize_punctuation = normalize_punctuation
        self.remove_zwj = remove_zwj

    def normalize(self, text: str) -> str:
        """Normalize the input Bangla text."""
        # Unicode NFC normalization first
        text = unicodedata.normalize("NFC", text)
        # Character substitutions
        if self.normalize_punctuation or self.remove_zwj:
            for src, tgt in self.CHAR_MAP.items():
                text = text.replace(src, tgt)
        # Digit conversion
        if self.normalize_digits:
            text = text.translate(ARABIC_TO_BANGLA)
        return text

    def __call__(self, text: str) -> str:
        return self.normalize(text)


class BanglaStopwordRemover:
    """
    Remove Bangla stopwords from tokenized or raw text.

    Parameters
    ----------
    custom_stopwords : list, optional
        Additional stopwords to include.
    keep_default : bool
        Whether to keep the built-in stopword list (default True).

    Examples
    --------
    >>> remover = BanglaStopwordRemover()
    >>> remover.remove_from_text("আমি এবং তুমি বাজারে গেলাম")
    'বাজারে গেলাম'
    """

    def __init__(
        self,
        custom_stopwords: Optional[List[str]] = None,
        keep_default: bool = True,
    ):
        self.stopwords: set = BANGLA_STOPWORDS.copy() if keep_default else set()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

    def remove(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from a list of tokens."""
        return [t for t in tokens if t not in self.stopwords]

    def remove_from_text(self, text: str) -> str:
        """Remove stopwords directly from a raw text string."""
        tokens = text.split()
        filtered = self.remove(tokens)
        return " ".join(filtered)

    def add_stopwords(self, words: List[str]) -> None:
        """Add new stopwords dynamically."""
        self.stopwords.update(words)

    def __call__(self, tokens: List[str]) -> List[str]:
        return self.remove(tokens)


class BanglaPunctuationHandler:
    """
    Handle Bangla-specific punctuation including daari (।), double-daari (॥),
    and common punctuation marks.

    Parameters
    ----------
    mode : str
        'remove' to strip all punctuation, 'normalize' to standardize,
        or 'split' to add spaces around punctuation marks (default 'normalize').

    Examples
    --------
    >>> handler = BanglaPunctuationHandler(mode='normalize')
    >>> handler.process("বাংলাদেশ।ভারত")
    'বাংলাদেশ। ভারত'
    """

    BANGLA_PUNCT = "।॥"
    ALL_PUNCT = r"[।॥,;:!?\"'()\[\]{}\-–—/\\@#$%^&*+=<>|~`]"

    def __init__(self, mode: str = "normalize"):
        assert mode in ("remove", "normalize", "split"), \
            "mode must be 'remove', 'normalize', or 'split'"
        self.mode = mode

    def process(self, text: str) -> str:
        if self.mode == "remove":
            return re.sub(self.ALL_PUNCT, " ", text).strip()
        elif self.mode == "normalize":
            # Ensure space after sentence-ending punctuation
            text = re.sub(r"([।॥!?])([^\s])", r"\1 \2", text)
            return text
        else:  # split
            text = re.sub(f"({self.ALL_PUNCT})", r" \1 ", text)
            return re.sub(r"\s+", " ", text).strip()

    def __call__(self, text: str) -> str:
        return self.process(text)


@dataclass
class BanglaPipeline:
    """
    Composable preprocessing pipeline for Bangla text.

    Chain multiple preprocessing steps together. Each step must be callable
    and accept a string, returning a string.

    Parameters
    ----------
    steps : list of callables
        Ordered list of preprocessing functions / objects.

    Examples
    --------
    >>> pipeline = BanglaPipeline(steps=[
    ...     BanglaTextCleaner(remove_urls=True),
    ...     BanglaNormalizer(normalize_digits=True),
    ...     BanglaPunctuationHandler(mode='normalize'),
    ... ])
    >>> pipeline.run("আমাদের ওয়েবসাইট https://example.com দেখুন।")
    'আমাদের ওয়েবসাইট  দেখুন।'
    """

    steps: List[Callable] = field(default_factory=list)

    def add_step(self, step: Callable) -> "BanglaPipeline":
        """Add a processing step and return self for chaining."""
        self.steps.append(step)
        return self

    def run(self, text: str) -> str:
        """Execute the full pipeline on a single text."""
        for step in self.steps:
            text = step(text)
        return text

    def run_batch(self, texts: List[str]) -> List[str]:
        """Execute the full pipeline on a list of texts."""
        return [self.run(t) for t in texts]

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(text, list):
            return self.run_batch(text)
        return self.run(text)

    def __repr__(self):
        step_names = [type(s).__name__ for s in self.steps]
        return f"BanglaPipeline(steps={step_names})"


__all__ = [
    "BanglaTextCleaner",
    "BanglaNormalizer",
    "BanglaStopwordRemover",
    "BanglaPunctuationHandler",
    "BanglaPipeline",
    "BANGLA_STOPWORDS",
]
