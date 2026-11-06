"""
Machine Learning Classifiers for Sentiment Analysis and Star Rating Prediction.

Includes:
- Multinomial Naive Bayes (with Laplace smoothing and log-probabilities)
- Logistic Regression / Linear Classifier
- Star Rating Classifier (1-5 stars)
- Save and load capability (JSON / Joblib)
"""
import json
import math
from typing import List, Dict, Tuple, Any, Optional

try:
    from sklearn.naive_bayes import MultinomialNB as SklearnNB
    from sklearn.linear_model import LogisticRegression as SklearnLogReg
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class PureNaiveBayes:
    """Zero-dependency Multinomial Naive Bayes Classifier."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Laplace smoothing
        self.classes: List[str] = []
        self.class_priors: Dict[str, float] = {}
        self.feature_log_prob: Dict[str, List[float]] = {}
        self.n_features: int = 0

    def fit(self, X: List[List[float]], y: List[str]):
        """Fits Naive Bayes from feature vectors and labels."""
        self.classes = sorted(list(set(y)))
        n_samples = len(y)
        self.n_features = len(X[0]) if n_samples > 0 else 0

        # Group samples by class
        class_counts = {c: 0 for c in self.classes}
        feature_sums = {c: [0.0] * self.n_features for c in self.classes}

        for row, label in zip(X, y):
            class_counts[label] += 1
            for j in range(self.n_features):
                feature_sums[label][j] += row[j]

        # Calculate priors and conditional probabilities with Laplace smoothing
        self.class_priors = {
            c: math.log(class_counts[c] / n_samples) for c in self.classes
        }

        self.feature_log_prob = {}
        for c in self.classes:
            total_feature_mass = sum(feature_sums[c]) + (self.alpha * self.n_features)
            log_probs = []
            for j in range(self.n_features):
                smoothed_prob = (feature_sums[c][j] + self.alpha) / total_feature_mass
                log_probs.append(math.log(smoothed_prob))
            self.feature_log_prob[c] = log_probs

        return self

    def predict_log_proba(self, X: List[List[float]]) -> List[Dict[str, float]]:
        """Returns unnormalized log-posterior score for each class."""
        predictions = []
        for row in X:
            scores = {}
            for c in self.classes:
                score = self.class_priors[c]
                log_probs = self.feature_log_prob[c]
                for j in range(self.n_features):
                    if row[j] > 0:
                        score += row[j] * log_probs[j]
                scores[c] = score
            predictions.append(scores)
        return predictions

    def predict_proba(self, X: List[List[float]]) -> List[Dict[str, float]]:
        """Returns normalized probability distribution using softmax."""
        log_scores = self.predict_log_proba(X)
        probabilities = []
        for scores in log_scores:
            max_s = max(scores.values())
            exp_scores = {c: math.exp(s - max_s) for c, s in scores.items()}
            total = sum(exp_scores.values())
            probs = {c: exp_scores[c] / (total if total > 0 else 1.0) for c in scores}
            probabilities.append(probs)
        return probabilities

    def predict(self, X: List[List[float]]) -> List[str]:
        """Predicts class with highest probability."""
        log_scores = self.predict_log_proba(X)
        return [max(s.items(), key=lambda item: item[1])[0] for s in log_scores]


class SentimentModel:
    """Wrapper supporting multiple ML algorithms with training and prediction."""

    def __init__(self, model_type: str = "naive_bayes"):
        self.model_type = model_type
        if model_type == "naive_bayes":
            self.model = PureNaiveBayes()
        elif model_type == "logistic_regression" and HAS_SKLEARN:
            self.model = SklearnLogReg(max_iter=500, random_state=42)
        else:
            self.model = PureNaiveBayes()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X) -> List[str]:
        preds = self.model.predict(X)
        return list(preds)

    def predict_proba(self, X) -> List[Dict[str, float]]:
        if hasattr(self.model, "predict_proba"):
            res = self.model.predict_proba(X)
            if isinstance(res, list) and isinstance(res[0], dict):
                return res
            # Convert sklearn ndarray to dicts
            classes = self.model.classes_
            return [{classes[i]: float(prob) for i, prob in enumerate(row)} for row in res]
        return []

    def save(self, filepath: str, feature_names: List[str]):
        """Serializes model parameters and vocabulary to JSON."""
        if isinstance(self.model, PureNaiveBayes):
            data = {
                "model_type": "pure_naive_bayes",
                "classes": self.model.classes,
                "class_priors": self.model.class_priors,
                "feature_log_prob": self.model.feature_log_prob,
                "n_features": self.model.n_features,
                "feature_names": feature_names
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def load(self, filepath: str) -> List[str]:
        """Loads serialized model parameters."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.model = PureNaiveBayes()
        self.model.classes = data["classes"]
        self.model.class_priors = data["class_priors"]
        self.model.feature_log_prob = data["feature_log_prob"]
        self.model.n_features = data["n_features"]
        return data.get("feature_names", [])


class StarRatingPredictor:
    """Predicts product star ratings (1 to 5) from review text."""

    def __init__(self):
        self.model = PureNaiveBayes()
        self.rating_classes = ["1", "2", "3", "4", "5"]

    def fit(self, X: List[List[float]], y_stars: List[int]):
        str_labels = [str(int(s)) for s in y_stars]
        self.model.fit(X, str_labels)
        return self

    def predict(self, X: List[List[float]]) -> List[int]:
        preds = self.model.predict(X)
        return [int(p) for p in preds]

    def predict_expected_rating(self, X: List[List[float]]) -> List[float]:
        """Calculates expected star rating as weighted expectation sum(star * p(star))."""
        probs_list = self.model.predict_proba(X)
        expected = []
        for p_dict in probs_list:
            weighted_sum = sum(int(star) * p_dict.get(star, 0.0) for star in self.rating_classes)
            expected.append(round(weighted_sum, 1))
        return expected

    def save(self, filepath: str, feature_names: List[str]):
        data = {
            "model_type": "star_rating_predictor",
            "classes": self.model.classes,
            "class_priors": self.model.class_priors,
            "feature_log_prob": self.model.feature_log_prob,
            "n_features": self.model.n_features,
            "feature_names": feature_names
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> List[str]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.model = PureNaiveBayes()
        self.model.classes = data["classes"]
        self.model.class_priors = data["class_priors"]
        self.model.feature_log_prob = data["feature_log_prob"]
        self.model.n_features = data["n_features"]
        return data.get("feature_names", [])
