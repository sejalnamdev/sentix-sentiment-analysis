"""
Unit and Integration Tests for Sentix Sentiment Analysis Pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.preprocessor import TextPreprocessor, simple_stem
from src.feature_extraction import ReviewVectorizer, PureTfidfVectorizer
from src.ml_models import PureNaiveBayes, SentimentModel, StarRatingPredictor
from src.evaluator import ModelEvaluator, train_test_split
from src.aspect_analyzer import AspectSentimentAnalyzer
from src.llm_engine import LLMSentimentAnalyzer


class TestSentixPipeline(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor(keep_negations=True, use_stemming=True)
        self.aspect_analyzer = AspectSentimentAnalyzer()

    def test_preprocessor_negation_retention(self):
        text = "This phone is not good, I don't like it."
        tokens = self.preprocessor.process(text)
        self.assertIn("not", tokens)
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)

    def test_stemming(self):
        self.assertEqual(simple_stem("running"), "runn")
        self.assertEqual(simple_stem("batteries"), "battery")

    def test_pure_tfidf_vectorizer(self):
        docs = ["battery life is amazing", "battery died quickly and terrible customer service"]
        vec = PureTfidfVectorizer(max_features=50)
        matrix = vec.fit_transform(docs)
        self.assertEqual(len(matrix), 2)
        self.assertGreater(len(vec.vocabulary_), 0)

    def test_naive_bayes_classification(self):
        X = [
            [1.0, 0.0, 1.0],
            [0.9, 0.1, 0.8],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.2]
        ]
        y = ["positive", "positive", "negative", "negative"]
        nb = PureNaiveBayes()
        nb.fit(X, y)
        preds = nb.predict([[0.8, 0.0, 0.9], [0.0, 0.9, 0.1]])
        self.assertEqual(preds, ["positive", "negative"])

    def test_aspect_based_sentiment_extraction(self):
        review = "The sound quality is fantastic, but the battery life is terrible."
        res = self.aspect_analyzer.analyze_review(review)
        self.assertTrue(res["quality"]["mentioned"])
        self.assertEqual(res["quality"]["sentiment"], "positive")
        self.assertTrue(res["battery"]["mentioned"])
        self.assertEqual(res["battery"]["sentiment"], "negative")

    def test_evaluator_metrics(self):
        y_true = ["positive", "positive", "negative", "negative"]
        y_pred = ["positive", "negative", "negative", "negative"]
        report = ModelEvaluator.classification_report(y_true, y_pred)
        self.assertEqual(report["accuracy"], 0.75)
        self.assertIn("positive", report)
        self.assertIn("negative", report)

    def test_llm_sarcasm_heuristic(self):
        engine = LLMSentimentAnalyzer()
        res = engine.analyze_review("Great phone if you enjoy charging it 5 times a day and having it broken.")
        self.assertTrue(res["is_sarcastic"])
        self.assertEqual(res["overall_sentiment"], "negative")


if __name__ == "__main__":
    unittest.main()
