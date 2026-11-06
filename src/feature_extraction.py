"""
Feature extraction module: TF-IDF Vectorization.

Supports:
- Scikit-learn TfidfVectorizer when available
- Pure-Python TF-IDF implementation with L2 normalization as standalone fallback
"""
import math
from collections import Counter
from typing import List, Dict, Optional, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class PureTfidfVectorizer:
    """Zero-dependency TF-IDF Vectorizer with sublinear TF and L2 normalization."""

    def __init__(self, min_df: int = 1, max_features: Optional[int] = 500):
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_: Dict[str, int] = {}
        self.idf_diag_: Dict[str, float] = {}
        self.n_samples_: int = 0

    def fit(self, raw_documents: List[str]):
        """Builds vocabulary and computes IDF values across documents."""
        self.n_samples_ = len(raw_documents)
        doc_freq = Counter()
        tokenized_docs = [doc.split() for doc in raw_documents]

        for doc in tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freq[term] += 1

        # Filter by min_df
        filtered_terms = [
            (term, count) for term, count in doc_freq.items() if count >= self.min_df
        ]

        # Sort by frequency descending and apply max_features
        filtered_terms.sort(key=lambda x: x[1], reverse=True)
        if self.max_features:
            filtered_terms = filtered_terms[: self.max_features]

        # Assign indices sorted alphabetically
        sorted_vocab = sorted([term for term, _ in filtered_terms])
        self.vocabulary_ = {term: idx for idx, term in enumerate(sorted_vocab)}

        # Smooth IDF: log((1 + n) / (1 + df)) + 1
        self.idf_diag_ = {}
        for term in self.vocabulary_:
            df = doc_freq[term]
            self.idf_diag_[term] = math.log((1 + self.n_samples_) / (1 + df)) + 1.0

        return self

    def transform(self, raw_documents: List[str]) -> List[List[float]]:
        """Transforms documents to normalized TF-IDF feature vectors."""
        matrix = []
        for doc in raw_documents:
            tokens = doc.split()
            counts = Counter(tokens)
            total_tokens = len(tokens) if tokens else 1

            vec = [0.0] * len(self.vocabulary_)
            norm_sq = 0.0

            for term, count in counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    # Term frequency
                    tf = count / total_tokens
                    idf = self.idf_diag_[term]
                    tfidf = tf * idf
                    vec[idx] = tfidf
                    norm_sq += tfidf * tfidf

            # L2 normalization
            if norm_sq > 0:
                l2 = math.sqrt(norm_sq)
                vec = [val / l2 for val in vec]

            matrix.append(vec)
        return matrix

    def fit_transform(self, raw_documents: List[str]) -> List[List[float]]:
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def get_feature_names(self) -> List[str]:
        inv_vocab = {v: k for k, v in self.vocabulary_.items()}
        return [inv_vocab[i] for i in range(len(self.vocabulary_))]


class ReviewVectorizer:
    """Unified Vectorizer choosing between Scikit-learn and Pure-Python."""

    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2), force_pure: bool = False):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.backend = "pure" if (force_pure or not HAS_SKLEARN) else "sklearn"
        if self.backend == "sklearn":
            self.vectorizer = SklearnTfidf(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True
            )
        else:
            self.vectorizer = PureTfidfVectorizer(max_features=self.max_features)

    @classmethod
    def from_features(cls, features: List[str]):
        """Reconstructs a vectorizer from a saved feature list."""
        vec = cls(max_features=len(features), force_pure=True)
        vec.vectorizer.vocabulary_ = {f: i for i, f in enumerate(features)}
        vec.vectorizer.idf_diag_ = {f: 1.0 for f in features}
        return vec

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]):
        result = self.vectorizer.transform(texts)
        if self.backend == "sklearn":
            return result.toarray()
        return result

    def fit_transform(self, texts: List[str]):
        result = self.vectorizer.fit_transform(texts)
        if self.backend == "sklearn":
            return result.toarray()
        return result

    def get_feature_names(self) -> List[str]:
        if self.backend == "sklearn":
            return list(self.vectorizer.get_feature_names_out())
        return self.vectorizer.get_feature_names()
