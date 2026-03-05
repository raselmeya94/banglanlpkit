"""
banglanlp.tokenization
=======================
Tokenization utilities for Bangla text including word, sentence,
subword (BPE), and character-level tokenizers.
"""

import re
from typing import List, Optional, Dict, Tuple, Union


# Bangla sentence boundary markers
SENTENCE_BOUNDARIES = re.compile(r"(?<=[।॥?!])\s+")
WORD_TOKENIZE_PATTERN = re.compile(r"[\u0980-\u09FF]+|[a-zA-Z0-9]+|[।॥,.?!;:\"\'\(\)\-]")


class BanglaWordTokenizer:
    """
    Word-level tokenizer for Bangla text.

    Handles Bangla conjunct consonants (যুক্তবর্ণ) and correctly
    segments words without breaking Unicode clusters.

    Parameters
    ----------
    split_punctuation : bool
        Whether to treat punctuation as separate tokens (default True).
    keep_english : bool
        Whether to preserve English words as tokens (default True).

    Examples
    --------
    >>> tokenizer = BanglaWordTokenizer()
    >>> tokenizer.tokenize("আমি বাংলায় কথা বলি।")
    ['আমি', 'বাংলায়', 'কথা', 'বলি', '।']
    """

    def __init__(
        self,
        split_punctuation: bool = True,
        keep_english: bool = True,
    ):
        self.split_punctuation = split_punctuation
        self.keep_english = keep_english

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        tokens = WORD_TOKENIZE_PATTERN.findall(text)
        if not self.split_punctuation:
            tokens = [t for t in tokens if not re.match(r"[^\u0980-\u09FFa-zA-Z0-9]", t)]
        if not self.keep_english:
            tokens = [t for t in tokens if not re.match(r"[a-zA-Z]", t)]
        return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a list of texts."""
        return [self.tokenize(t) for t in texts]

    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to a string."""
        text = " ".join(tokens)
        # Fix spacing around punctuation
        text = re.sub(r"\s([।॥,.?!;:])", r"\1", text)
        return text

    def __call__(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        if isinstance(text, list):
            return self.tokenize_batch(text)
        return self.tokenize(text)

    def __repr__(self):
        return f"BanglaWordTokenizer(split_punctuation={self.split_punctuation})"


class BanglaSentenceTokenizer:
    """
    Sentence-level tokenizer for Bangla text.

    Splits text on Bangla daari (।), double-daari (॥), and
    standard punctuation while handling abbreviations.

    Parameters
    ----------
    min_sentence_length : int
        Minimum character length for a valid sentence (default 3).

    Examples
    --------
    >>> tokenizer = BanglaSentenceTokenizer()
    >>> tokenizer.tokenize("আমি ঢাকায় থাকি। সে চট্টগ্রামে থাকে।")
    ['আমি ঢাকায় থাকি।', 'সে চট্টগ্রামে থাকে।']
    """

    SENTENCE_END = re.compile(r"(?<=[।॥!?])\s*")

    def __init__(self, min_sentence_length: int = 3):
        self.min_sentence_length = min_sentence_length

    def tokenize(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Ensure space after sentence markers
        text = re.sub(r"([।॥!?])(?=[^\s])", r"\1 ", text)
        sentences = self.SENTENCE_END.split(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= self.min_sentence_length]
        return sentences

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)


class BanglaSubwordTokenizer:
    """
    Subword tokenizer for Bangla using a vocabulary-based approach.

    Suitable for neural models. Learns subword units from a corpus or
    loads from a pretrained vocabulary file.

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size (default 32000).
    model_path : str, optional
        Path to a pretrained SentencePiece or BPE model file.

    Examples
    --------
    >>> tokenizer = BanglaSubwordTokenizer(vocab_size=16000)
    >>> tokenizer.tokenize("বাংলাদেশের মানুষ")
    ['▁বাংলাদেশের', '▁মানুষ']
    """

    SPECIAL_TOKENS = {
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2,
        "[SEP]": 3, "[MASK]": 4,
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        model_path: Optional[str] = None,
    ):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self._model = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load a pretrained SentencePiece model."""
        try:
            import sentencepiece as spm
            self._model = spm.SentencePieceProcessor()
            self._model.Load(path)
        except ImportError:
            raise ImportError(
                "sentencepiece is required for BanglaSubwordTokenizer. "
                "Install with: pip install sentencepiece"
            )

    def train(self, corpus_path: str, output_prefix: str = "bangla_spm") -> None:
        """
        Train a SentencePiece model on a Bangla corpus.

        Parameters
        ----------
        corpus_path : str
            Path to a plain-text Bangla corpus file.
        output_prefix : str
            Prefix for the output model files.
        """
        try:
            import sentencepiece as spm
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=output_prefix,
                vocab_size=self.vocab_size,
                character_coverage=0.9999,
                model_type="bpe",
                pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            )
            self._load_model(f"{output_prefix}.model")
        except ImportError:
            raise ImportError("sentencepiece is required. pip install sentencepiece")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword pieces."""
        if self._model:
            return self._model.encode(text, out_type=str)
        # Fallback: character-level tokenization
        return list(text.replace(" ", "▁"))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self._model:
            return self._model.encode(text)
        return [ord(c) for c in text]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if self._model:
            return self._model.decode(ids)
        return "".join(chr(i) for i in ids)

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)


class BanglaBPETokenizer:
    """
    Byte-Pair Encoding tokenizer for Bangla, compatible with HuggingFace tokenizers.

    Parameters
    ----------
    pretrained_name : str, optional
        HuggingFace model name to load tokenizer from
        (e.g., 'sagorsarker/bangla-bert-base').

    Examples
    --------
    >>> tokenizer = BanglaBPETokenizer(pretrained_name='sagorsarker/bangla-bert-base')
    >>> tokenizer.tokenize("বাংলা একটি সুন্দর ভাষা")
    ['বাংলা', 'একটি', 'সুন্দর', 'ভাষা']
    """

    SUPPORTED_MODELS: Dict[str, str] = {
        "bangla-bert-base": "sagorsarker/bangla-bert-base",
        "banglabert": "csebuetnlp/banglabert",
        "muril": "google/muril-base-cased",
        "xlm-roberta": "xlm-roberta-base",
    }

    def __init__(self, pretrained_name: Optional[str] = None):
        self.pretrained_name = pretrained_name
        self._tokenizer = None
        if pretrained_name:
            self._load_pretrained(pretrained_name)

    def _load_pretrained(self, name: str) -> None:
        model_name = self.SUPPORTED_MODELS.get(name, name)
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ImportError:
            raise ImportError("transformers is required. pip install transformers")

    def tokenize(self, text: str) -> List[str]:
        if self._tokenizer:
            return self._tokenizer.tokenize(text)
        raise RuntimeError("No pretrained model loaded. Pass pretrained_name to constructor.")

    def encode(self, text: str, return_tensors: Optional[str] = None) -> Union[List[int], "torch.Tensor"]:
        if self._tokenizer:
            return self._tokenizer.encode(text, return_tensors=return_tensors)
        raise RuntimeError("No pretrained model loaded.")

    def batch_encode(self, texts: List[str], **kwargs) -> Dict:
        if self._tokenizer:
            return self._tokenizer(texts, **kwargs)
        raise RuntimeError("No pretrained model loaded.")

    @classmethod
    def list_supported_models(cls) -> List[str]:
        """List all built-in supported pretrained model shortcuts."""
        return list(cls.SUPPORTED_MODELS.keys())

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)

    def __repr__(self):
        return f"BanglaBPETokenizer(pretrained_name='{self.pretrained_name}')"


__all__ = [
    "BanglaWordTokenizer",
    "BanglaSentenceTokenizer",
    "BanglaSubwordTokenizer",
    "BanglaBPETokenizer",
]
