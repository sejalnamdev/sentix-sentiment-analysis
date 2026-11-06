"""
Batch Processing Pipeline for Product Reviews CSV.

Features:
- Processes CSV reviews in bulk
- Applies sentiment classification and star rating prediction
- Extracts aspect-level sentiments (Battery, Quality, Price, Delivery, Usability)
- Extracts top positive & negative keywords/n-grams
- Calculates aggregate metrics and exports enriched CSV dataset
"""
import csv
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple
from .preprocessor import TextPreprocessor
from .aspect_analyzer import AspectSentimentAnalyzer
from .llm_engine import LLMSentimentAnalyzer


class BatchReviewProcessor:
    """Processes datasets of customer reviews and generates aggregate analytics."""

    def __init__(self, ml_model=None, vectorizer=None, star_model=None, use_llm: bool = False):
        self.preprocessor = TextPreprocessor()
        self.aspect_analyzer = AspectSentimentAnalyzer()
        self.llm_analyzer = LLMSentimentAnalyzer() if use_llm else None
        self.ml_model = ml_model
        self.vectorizer = vectorizer
        self.star_model = star_model

    def read_csv(self, filepath: str) -> List[Dict[str, str]]:
        """Reads CSV file handling varied header names."""
        rows = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def process_reviews(self, reviews: List[Dict[str, str]]) -> Dict[str, Any]:
        """Processes each review in the list and computes analytics."""
        processed_rows = []
        sentiment_counts = Counter()
        aspect_aggregates = {
            "battery": {"positive": 0, "negative": 0, "neutral": 0, "total": 0},
            "quality": {"positive": 0, "negative": 0, "neutral": 0, "total": 0},
            "price": {"positive": 0, "negative": 0, "neutral": 0, "total": 0},
            "delivery": {"positive": 0, "negative": 0, "neutral": 0, "total": 0},
            "usability": {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
        }
        pos_keywords = Counter()
        neg_keywords = Counter()
        star_ratings = []

        # Vectorize all reviews if ML model is provided
        review_texts = [r.get("review_body") or r.get("review_text") or r.get("text") or r.get("review", "") for r in reviews]
        processed_texts = [self.preprocessor.get_processed_string(t) for t in review_texts]

        ml_preds = None
        ml_stars = None
        if self.ml_model and self.vectorizer:
            feature_matrix = self.vectorizer.transform(processed_texts)
            ml_preds = self.ml_model.predict(feature_matrix)
            if self.star_model:
                ml_stars = self.star_model.predict(feature_matrix)

        for i, row in enumerate(reviews):
            raw_text = review_texts[i]
            row_id = row.get("review_id", str(i + 1))
            asin = row.get("asin", f"B0{i:08d}")
            product_name = row.get("product_name", "Amazon Product")
            product_category = row.get("product_category", "General")
            review_headline = row.get("review_headline", "")
            verified_purchase = row.get("verified_purchase", "Y")

            # 1. Determine sentiment and star rating
            if self.llm_analyzer:
                llm_res = self.llm_analyzer.analyze_review(raw_text)
                sentiment = llm_res.get("overall_sentiment", "neutral")
                predicted_stars = llm_res.get("predicted_star_rating", 3)
                is_sarcastic = llm_res.get("is_sarcastic", False)
                aspect_data = llm_res.get("aspects", {})
            else:
                if ml_preds:
                    sentiment = ml_preds[i]
                    predicted_stars = ml_stars[i] if ml_stars else (5 if sentiment == "positive" else 1)
                else:
                    sentiment = "neutral"
                    predicted_stars = 3
                is_sarcastic = False
                aspect_data = self.aspect_analyzer.analyze_review(raw_text)

            sentiment_counts[sentiment] += 1
            star_ratings.append(predicted_stars)

            # 2. Aspect counts
            mentioned_aspects = []
            aspect_dict = aspect_data.get("aspects", aspect_data) if isinstance(aspect_data, dict) else {}
            for asp, d in aspect_dict.items():
                if isinstance(d, dict) and d.get("mentioned", False):
                    asp_sent = d.get("sentiment", "neutral")
                    if asp in aspect_aggregates:
                        aspect_aggregates[asp]["total"] += 1
                        if asp_sent in aspect_aggregates[asp]:
                            aspect_aggregates[asp][asp_sent] += 1
                    mentioned_aspects.append(f"{asp}:{asp_sent}")

            # 3. Top tokens
            tokens = self.preprocessor.process(raw_text)
            if sentiment == "positive":
                pos_keywords.update(tokens)
            elif sentiment == "negative":
                neg_keywords.update(tokens)

            processed_rows.append({
                "review_id": row_id,
                "asin": asin,
                "product_category": product_category,
                "product_name": product_name,
                "review_headline": review_headline,
                "review_text": raw_text,
                "predicted_sentiment": sentiment,
                "predicted_star_rating": predicted_stars,
                "is_sarcastic": is_sarcastic,
                "verified_purchase": verified_purchase,
                "detected_aspects": "; ".join(mentioned_aspects)
            })

        total = len(processed_rows) if processed_rows else 1
        avg_star = sum(star_ratings) / total if star_ratings else 0.0

        return {
            "total_reviews": len(processed_rows),
            "sentiment_distribution": dict(sentiment_counts),
            "sentiment_percentages": {
                k: round((v / total) * 100, 1) for k, v in sentiment_counts.items()
            },
            "average_star_rating": round(avg_star, 2),
            "aspect_breakdown": aspect_aggregates,
            "top_positive_keywords": pos_keywords.most_common(10),
            "top_negative_keywords": neg_keywords.most_common(10),
            "reviews": processed_rows
        }

    def export_csv(self, processed_rows: List[Dict[str, Any]], output_filepath: str):
        """Exports enriched review dataset to CSV."""
        if not processed_rows:
            return
        fields = list(processed_rows[0].keys())
        with open(output_filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(processed_rows)
