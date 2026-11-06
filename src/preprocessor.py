"""
Text Preprocessing Pipeline for Product Review Sentiment Analysis.

Handles:
- Contraction expansion (preserving negations like "don't" -> "do not")
- Negation-aware tokenization
- Stopword filtering with negation retention
- Rule-based lemmatization/stemming
- N-gram generation (unigrams + bigrams)
"""
import re
from typing import List, Tuple
from .config import STOPWORDS, NEGATION_WORDS

# Common English contractions
CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bn't\b": " not",
    r"\b're\b": " are",
    r"\b's\b": " is",
    r"\b'd\b": " would",
    r"\b'll\b": " will",
    r"\b't\b": " not",
    r"\b've\b": " have",
    r"\b'm\b": " am"
}

# Rule-based suffix stripping (lightweight, zero-dependency Porter-style stemmer)
def simple_stem(word: str) -> str:
    """Applies basic morphological normalization to reduce inflected words."""
    if len(word) <= 3:
        return word
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ss"):
        return word
    if word.endswith("s") and not word.endswith("us") and not word.endswith("is"):
        return word[:-1]
    if word.endswith("eed"):
        return word[:-1]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    return word


class TextPreprocessor:
    """Preprocesses raw review text into clean normalized tokens and n-grams."""

    def __init__(self, keep_negations: bool = True, use_stemming: bool = True):
        self.keep_negations = keep_negations
        self.use_stemming = use_stemming
        # Stopwords minus negations if keep_negations is True
        if self.keep_negations:
            self.active_stopwords = STOPWORDS - NEGATION_WORDS
        else:
            self.active_stopwords = STOPWORDS

    def clean_text(self, text: str) -> str:
        """Cleans and standardizes raw text."""
        if not text:
            return ""
        text = text.lower()
        # Expand contractions
        for pattern, replacement in CONTRACTIONS.items():
            text = re.sub(pattern, replacement, text)
        # Remove URLs and email addresses
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        # Keep alphanumeric, apostrophes, and spaces; strip other symbols
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Collapse multi-spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Splits clean text into tokens."""
        cleaned = self.clean_text(text)
        return cleaned.split()

    def process(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline:
        Cleaning -> Tokenizing -> Stopword Filtering (Negation-aware) -> Stemming.
        """
        tokens = self.tokenize(text)
        filtered = []
        for token in tokens:
            if token in self.active_stopwords:
                continue
            if len(token) <= 1 and token not in ("a", "i"):
                continue
            stemmed = simple_stem(token) if self.use_stemming else token
            filtered.append(stemmed)
        return filtered

    def get_processed_string(self, text: str) -> str:
        """Returns space-separated processed tokens for vectorizers."""
        return " ".join(self.process(text))

    def extract_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Generates n-grams (e.g. bigrams) from token sequence."""
        if len(tokens) < n:
            return []
        return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def extract_unigrams_and_bigrams(self, text: str) -> List[str]:
        """Combines unigrams and bigrams for enhanced sentiment context."""
        tokens = self.process(text)
        bigrams = self.extract_ngrams(tokens, n=2)
        return tokens + bigrams
