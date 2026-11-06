"""
Sentix - Master Command Line Interface & Runner.

Usage:
    # 1. Full end-to-end training, evaluation & dashboard generation
    python run_pipeline.py

    # 2. Train and evaluate ML models
    python run_pipeline.py --train

    # 3. Analyze a single custom review (ML + Aspect Analysis)
    python run_pipeline.py --analyze "The camera is stunning, but the battery drains in 3 hours."

    # 4. Analyze using Gemini LLM reasoning & sarcasm detection
    python run_pipeline.py --analyze "Great phone if you enjoy charging it 5 times a day!" --llm

    # 5. Process a batch CSV file and produce interactive dashboard
    python run_pipeline.py --batch data/test_reviews.csv
"""
import argparse
import csv
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows command line
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.config import (
    DEFAULT_TRAIN_DATA,
    DEFAULT_TEST_DATA,
    MODEL_SAVE_PATH,
    STAR_MODEL_SAVE_PATH,
    DASHBOARD_OUTPUT_PATH,
    OUTPUT_DIR
)
from src.preprocessor import TextPreprocessor
from src.feature_extraction import ReviewVectorizer
from src.ml_models import SentimentModel, StarRatingPredictor
from src.evaluator import ModelEvaluator, train_test_split
from src.aspect_analyzer import AspectSentimentAnalyzer
from src.llm_engine import LLMSentimentAnalyzer
from src.batch_pipeline import BatchReviewProcessor
from src.visualizer import generate_html_dashboard


def load_dataset(csv_path: Path):
    """Loads CSV reviews dataset supporting Amazon and standard formats."""
    texts = []
    sentiments = []
    stars = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = row.get("review_body") or row.get("review_text") or row.get("text") or row.get("review", "")
            sent = row.get("sentiment", "").strip().lower()
            star_val = row.get("star_rating") or row.get("overall") or row.get("rating", "3")
            try:
                star = int(float(star_val))
            except Exception:
                star = 3
            if not sent:
                sent = "positive" if star >= 4 else ("negative" if star <= 2 else "neutral")

            if txt:
                texts.append(txt)
                sentiments.append(sent)
                stars.append(star)
    return texts, sentiments, stars


def run_training():
    """Trains Sentiment and Star Rating ML models and outputs evaluation metrics."""
    print("=" * 70)
    print("[SENTIX] MACHINE LEARNING TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    print(f"[*] Loading training dataset from: {DEFAULT_TRAIN_DATA}")
    texts, sentiments, stars = load_dataset(DEFAULT_TRAIN_DATA)
    print(f"[*] Total dataset size: {len(texts)} product reviews.")

    # 1. Preprocessing
    print("[*] Preprocessing review texts (cleaning, negation retention, stemming)...")
    preprocessor = TextPreprocessor(keep_negations=True, use_stemming=True)
    clean_texts = [preprocessor.get_processed_string(t) for t in texts]

    # 2. Train-Test Split (80% train, 20% test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        clean_texts, sentiments, test_size=0.25, random_seed=42
    )
    _, _, y_train_stars, y_test_stars = train_test_split(
        clean_texts, stars, test_size=0.25, random_seed=42
    )
    print(f"[*] Train set: {len(X_train_raw)} samples | Test set: {len(X_test_raw)} samples.")

    # 3. TF-IDF Feature Extraction
    print("[*] Extracting TF-IDF feature vectors (unigrams + bigrams)...")
    vectorizer = ReviewVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    feature_names = vectorizer.get_feature_names()
    print(f"[*] Vocabulary size: {len(feature_names)} features extracted.")

    # 4. Model Training: Sentiment Classifier
    print("[*] Training Multinomial Naive Bayes Sentiment Classifier...")
    sentiment_model = SentimentModel(model_type="naive_bayes")
    sentiment_model.fit(X_train_vec, y_train)

    # 5. Model Training: Star Rating Predictor
    print("[*] Training Star Rating Predictor (1 to 5 Stars)...")
    star_model = StarRatingPredictor()
    star_model.fit(X_train_vec, y_train_stars)

    # 6. Evaluation
    y_pred = sentiment_model.predict(X_test_vec)
    report = ModelEvaluator.classification_report(y_test, y_pred)
    labels, matrix = ModelEvaluator.confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 70)
    print("[RESULTS] MODEL EVALUATION METRICS (TEST SET)")
    print("=" * 70)
    print(ModelEvaluator.format_report_table(report))

    print("\nConfusion Matrix:")
    print(f"                Predicted {labels[0]:<10} Predicted {labels[1]:<10}")
    print(f"Actual {labels[0]:<8}: {matrix[0][0]:<20} {matrix[0][1]:<20}")
    print(f"Actual {labels[1]:<8}: {matrix[1][0]:<20} {matrix[1][1]:<20}")

    # Star rating evaluation
    y_pred_stars = star_model.predict(X_test_vec)
    exact_match = sum(1 for yt, yp in zip(y_test_stars, y_pred_stars) if yt == yp)
    within_one = sum(1 for yt, yp in zip(y_test_stars, y_pred_stars) if abs(yt - yp) <= 1)
    print(f"\nStar Rating Exact Accuracy: {exact_match / len(y_test_stars):.1%}")
    print(f"Star Rating Within +/- 1 Star: {within_one / len(y_test_stars):.1%}")

    # 7. Model Persistence
    print(f"\n[*] Saving trained model artifacts to {MODEL_SAVE_PATH} ...")
    sentiment_model.save(str(MODEL_SAVE_PATH), feature_names)
    star_model.save(str(STAR_MODEL_SAVE_PATH), feature_names)
    print("[✓] Model artifacts saved successfully.")

    return sentiment_model, vectorizer, star_model


def run_single_analysis(review_text: str, use_llm: bool = False):
    """Analyzes a single customer review with aspect and star prediction."""
    print("=" * 70)
    print(f"[ANALYSIS] REVIEW REVIEW ANALYSIS ({'GEMINI LLM' if use_llm else 'CLASSICAL NLP & ASPECT ENGINE'})")
    print("=" * 70)
    print(f"Input: \"{review_text}\"\n")

    if use_llm:
        engine = LLMSentimentAnalyzer()
        res = engine.analyze_review(review_text)
        print(f"Overall Sentiment : {res.get('overall_sentiment', '').upper()}")
        print(f"Confidence        : {res.get('confidence', 0.0):.1%}")
        print(f"Predicted Stars   : {'★' * res.get('predicted_star_rating', 1)} ({res.get('predicted_star_rating')} / 5)")
        print(f"Sarcasm Detected  : {'YES' if res.get('is_sarcastic') else 'No'}")
        if res.get("sarcasm_explanation"):
            print(f"Sarcasm Reason    : {res.get('sarcasm_explanation')}")
        print("\nAspect-Based Breakdown:")
        for aspect, data in res.get("aspects", {}).items():
            if data.get("mentioned"):
                status = data.get("sentiment", "neutral").upper()
                score = data.get("score", 0.0)
                reason = data.get("reason", "N/A")
                print(f"  • {aspect.capitalize():<12}: [{status}] (score: {score}) -> \"{reason}\"")
        print(f"\nSummary: {res.get('summary', '')}")
    else:
        # Rule & Aspect Analyzer
        preprocessor = TextPreprocessor()
        clean = preprocessor.get_processed_string(review_text)
        aspect_engine = AspectSentimentAnalyzer()
        aspect_res = aspect_engine.analyze_review(review_text)

        # Try loading ML model
        sentiment_label = "Neutral"
        if MODEL_SAVE_PATH.exists():
            model = SentimentModel()
            features = model.load(str(MODEL_SAVE_PATH))
            vec = ReviewVectorizer.from_features(features)
            vec_repr = vec.transform([clean])
            sentiment_label = model.predict(vec_repr)[0].upper()

        print(f"Predicted Sentiment : {sentiment_label}")
        print("\nAspect-Based Breakdown:")
        aspect_dict = aspect_res.get("aspects", {})
        for aspect, data in aspect_dict.items():
            if isinstance(data, dict) and data.get("mentioned"):
                st = data.get("sentiment", "neutral").upper()
                sc = data.get("score", 0.0)
                snip = data["snippets"][0] if data.get("snippets") else ""
                print(f"  • {aspect.capitalize():<12}: [{st}] (score: {sc:+.2f}) -> \"{snip}\"")
            else:
                print(f"  • {aspect.capitalize():<12}: [Not Mentioned]")


def run_batch_processing(csv_path: Path, use_llm: bool = False):
    """Processes batch CSV, generates enriched CSV and interactive dashboard."""
    print("=" * 70)
    print("[BATCH] BATCH REVIEW PROCESSING & ANALYTICS PIPELINE")
    print("=" * 70)
    print(f"[*] Ingesting reviews from: {csv_path}")

    # Ensure models are trained
    if not MODEL_SAVE_PATH.exists() and not use_llm:
        print("[*] Training models before batch processing...")
        sentiment_model, vectorizer, star_model = run_training()
    elif not use_llm:
        sentiment_model = SentimentModel()
        features = sentiment_model.load(str(MODEL_SAVE_PATH))
        star_model = StarRatingPredictor()
        star_model.load(str(STAR_MODEL_SAVE_PATH))
        vectorizer = ReviewVectorizer.from_features(features)
    else:
        sentiment_model = None
        vectorizer = None
        star_model = None

    processor = BatchReviewProcessor(
        ml_model=sentiment_model,
        vectorizer=vectorizer,
        star_model=star_model,
        use_llm=use_llm
    )

    rows = processor.read_csv(str(csv_path))
    print(f"[*] Processing {len(rows)} reviews...")
    analytics = processor.process_reviews(rows)

    # Export enriched dataset
    output_csv = OUTPUT_DIR / "enriched_reviews.csv"
    processor.export_csv(analytics["reviews"], str(output_csv))
    print(f"[✓] Enriched dataset exported to: {output_csv}")

    # Generate interactive HTML dashboard
    generate_html_dashboard(analytics, str(DASHBOARD_OUTPUT_PATH))
    print(f"[✓] Dashboard viewable at: {DASHBOARD_OUTPUT_PATH}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("[METRICS] AGGREGATE BATCH METRICS")
    print("=" * 70)
    print(f"Total Reviews Analyzed : {analytics['total_reviews']}")
    print(f"Average Star Rating    : {analytics['average_star_rating']} / 5.0")
    print("Sentiment Distribution :")
    for s, count in analytics["sentiment_distribution"].items():
        pct = analytics["sentiment_percentages"].get(s, 0)
        print(f"  • {s.capitalize():<10}: {count} ({pct}%)")

    print("\nTop Positive Keywords  :", ", ".join([w for w, _ in analytics["top_positive_keywords"][:5]]))
    print("Top Negative Keywords  :", ", ".join([w for w, _ in analytics["top_negative_keywords"][:5]]))


def main():
    parser = argparse.ArgumentParser(description="Sentix - Product Review Sentiment Analysis System")
    parser.add_argument("--train", action="store_true", help="Train and evaluate ML models")
    parser.add_argument("--analyze", type=str, help="Analyze a single review string")
    parser.add_argument("--batch", type=str, help="Process a CSV review dataset")
    parser.add_argument("--llm", action="store_true", help="Enable Gemini LLM analysis")
    parser.add_argument("--clear-logs", action="store_true", help="Clear all stored execution and application logs")

    args = parser.parse_args()

    if args.clear_logs:
        log_dir = OUTPUT_DIR / "logs"
        if log_dir.exists():
            for f in log_dir.glob("*.log"):
                try:
                    f.unlink()
                except Exception:
                    pass
        print("[✓] All application and execution logs cleared successfully.")
        return

    if args.train:
        run_training()
    elif args.analyze:
        run_single_analysis(args.analyze, use_llm=args.llm)
    elif args.batch:
        run_batch_processing(Path(args.batch), use_llm=args.llm)
    else:
        # Full default execution
        run_training()
        print("\n")
        run_batch_processing(DEFAULT_TEST_DATA, use_llm=args.llm)
        print("\n")
        run_single_analysis("The sound is crystal clear and battery lasts forever, but customer support was awful.")


if __name__ == "__main__":
    main()
