"""
Sentix Enterprise - High-Performance Web Application & REST API Server.

Production Server for Amazon Product Review Sentiment Analysis & Aspect Intelligence.
"""
import csv
import http.server
import json
import socketserver
import sys
import urllib.parse
from pathlib import Path
from typing import Dict, Any

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
PORT = int(os.environ.get("PORT", 5000))
BASE_DIR = Path(__file__).resolve().parent

APP_STATE: Dict[str, Any] = {
    "reviews": [],
    "analytics": {},
    "ml_model": None,
    "vectorizer": None,
    "star_model": None
}


def initialize_app():
    """Initializes models and loads Amazon dataset."""
    from src.config import AMAZON_DATA_PATH, MODEL_SAVE_PATH, STAR_MODEL_SAVE_PATH
    from src.feature_extraction import ReviewVectorizer
    from src.ml_models import SentimentModel, StarRatingPredictor
    from src.batch_pipeline import BatchReviewProcessor
    from src.aspect_analyzer import AspectSentimentAnalyzer
    from src.llm_engine import LLMSentimentAnalyzer

    print("[*] Initializing Sentix Enterprise Review Analytics Engine...")

    if MODEL_SAVE_PATH.exists():
        model = SentimentModel()
        features = model.load(str(MODEL_SAVE_PATH))
        vec = ReviewVectorizer.from_features(features)
        APP_STATE["ml_model"] = model
        APP_STATE["vectorizer"] = vec

    if STAR_MODEL_SAVE_PATH.exists():
        star_m = StarRatingPredictor()
        star_m.load(str(STAR_MODEL_SAVE_PATH))
        APP_STATE["star_model"] = star_m

    APP_STATE["aspect_engine"] = AspectSentimentAnalyzer()
    APP_STATE["llm_engine"] = LLMSentimentAnalyzer()

    dataset_file = AMAZON_DATA_PATH if AMAZON_DATA_PATH.exists() else (BASE_DIR / "data" / "amazon_product_reviews.csv")
    processor = BatchReviewProcessor(
        ml_model=APP_STATE["ml_model"],
        vectorizer=APP_STATE["vectorizer"],
        star_model=APP_STATE["star_model"]
    )
    rows = processor.read_csv(str(dataset_file))
    analytics = processor.process_reviews(rows)
    APP_STATE["reviews"] = analytics["reviews"]
    APP_STATE["analytics"] = analytics
    print(f"[✓] Loaded {len(APP_STATE['reviews'])} Amazon product reviews across departments.")


class AppRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler with REST API and HTML UI."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.serve_ui()
        elif path == "/api/reviews":
            self.handle_api_reviews(parsed.query)
        elif path == "/api/metrics":
            self.handle_api_metrics()
        else:
            file_path = BASE_DIR / "output" / path.lstrip("/")
            if file_path.exists() and file_path.is_file():
                self.send_response(200)
                if file_path.suffix == ".html":
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                elif file_path.suffix == ".css":
                    self.send_header("Content-Type", "text/css")
                elif file_path.suffix == ".js":
                    self.send_header("Content-Type", "application/javascript")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/analyze":
            self.handle_api_analyze()
        elif path == "/api/reviews/write":
            self.handle_api_write_review()
        elif path == "/api/upload":
            self.handle_api_upload()
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def handle_api_reviews(self, query_string: str):
        params = urllib.parse.parse_qs(query_string)
        cat = params.get("category", [None])[0]
        sentiment = params.get("sentiment", [None])[0]
        aspect = params.get("aspect", [None])[0]
        search = (params.get("search", [""])[0]).lower()

        reviews = APP_STATE["reviews"]
        filtered = []
        for r in reviews:
            if cat and cat != "all" and r.get("product_category", "").lower() != cat.lower():
                continue
            if sentiment and sentiment != "all" and r.get("predicted_sentiment", "").lower() != sentiment.lower():
                continue
            if aspect and aspect != "all" and aspect.lower() not in r.get("detected_aspects", "").lower():
                continue
            if search:
                haystack = (
                    r.get("product_name", "") + " " +
                    r.get("review_headline", "") + " " +
                    r.get("review_text", "") + " " +
                    r.get("asin", "")
                ).lower()
                if search not in haystack:
                    continue
            filtered.append(r)

        self.send_json({
            "total_matches": len(filtered),
            "reviews": filtered
        })

    def handle_api_metrics(self):
        self.send_json(APP_STATE["analytics"])

    def handle_api_analyze(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            req_data = json.loads(body)
            review_text = (req_data.get("review_text") or req_data.get("review") or "").strip()
            use_llm = req_data.get("use_llm", False)

            if not review_text:
                self.send_json({"error": "Review text is empty"}, status=400)
                return

            if use_llm:
                result = APP_STATE["llm_engine"].analyze_review(review_text)
            else:
                aspect_res = APP_STATE["aspect_engine"].analyze_review(review_text)
                sentiment = aspect_res.get("overall_sentiment", "neutral")
                star = aspect_res.get("predicted_star_rating", 3)
                confidence = aspect_res.get("confidence", 0.90)
                is_sarcastic = aspect_res.get("is_sarcastic", False)
                sarcasm_exp = aspect_res.get("sarcasm_explanation", "")

                result = {
                    "overall_sentiment": sentiment,
                    "confidence": confidence,
                    "predicted_star_rating": star,
                    "is_sarcastic": is_sarcastic,
                    "sarcasm_explanation": sarcasm_exp,
                    "aspects": aspect_res.get("aspects", {}),
                    "summary": f"Review analyzed with {sentiment.upper()} polarity and {star}-star rating."
                }

            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_api_write_review(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            data = json.loads(body)
            product_name = data.get("product_name", "Amazon Product").strip()
            category = data.get("product_category", "Electronics").strip()
            headline = data.get("review_headline", "Customer Review").strip()
            text = data.get("review_body", "").strip()
            user_stars = int(data.get("star_rating", 5))

            if not text:
                self.send_json({"error": "Review body cannot be empty"}, status=400)
                return

            aspect_res = APP_STATE["aspect_engine"].analyze_review(text)
            sentiment = aspect_res.get("overall_sentiment", "positive" if user_stars >= 4 else "negative")
            detected_aspects = []
            for asp, d in aspect_res.get("aspects", {}).items():
                if d.get("mentioned"):
                    detected_aspects.append(f"{asp}:{d.get('sentiment', 'neutral')}")

            new_id = len(APP_STATE["reviews"]) + 1
            asin = f"B0{90000000 + new_id}"

            new_review = {
                "review_id": str(new_id),
                "asin": asin,
                "product_category": category,
                "product_name": product_name,
                "review_headline": headline,
                "review_text": text,
                "predicted_sentiment": sentiment,
                "predicted_star_rating": user_stars,
                "is_sarcastic": aspect_res.get("is_sarcastic", False),
                "verified_purchase": "Y",
                "detected_aspects": "; ".join(detected_aspects)
            }

            APP_STATE["reviews"].insert(0, new_review)

            dist = APP_STATE["analytics"].setdefault("sentiment_distribution", {"positive": 0, "negative": 0})
            dist[sentiment] = dist.get(sentiment, 0) + 1
            total = len(APP_STATE["reviews"])
            APP_STATE["analytics"]["total_reviews"] = total
            APP_STATE["analytics"]["sentiment_percentages"] = {
                k: round((v / total) * 100, 1) for k, v in dist.items()
            }

            self.send_json({
                "status": "success",
                "message": "Review published and analyzed successfully!",
                "new_review": new_review,
                "updated_metrics": APP_STATE["analytics"]
            })
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def handle_api_upload(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8", errors="replace")
        try:
            lines = body.splitlines()
            reader = csv.DictReader(lines)
            rows = [r for r in reader]

            from src.batch_pipeline import BatchReviewProcessor
            processor = BatchReviewProcessor(
                ml_model=APP_STATE["ml_model"],
                vectorizer=APP_STATE["vectorizer"],
                star_model=APP_STATE["star_model"]
            )
            analytics = processor.process_reviews(rows)
            APP_STATE["reviews"] = analytics["reviews"]
            APP_STATE["analytics"] = analytics

            self.send_json({
                "message": f"Successfully ingested {len(rows)} reviews",
                "analytics": analytics
            })
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)

    def serve_ui(self):
        dashboard_path = BASE_DIR / "output" / "sentiment_dashboard.html"
        if not dashboard_path.exists():
            from src.visualizer import generate_html_dashboard
            generate_html_dashboard(APP_STATE["analytics"], str(dashboard_path))

        with open(dashboard_path, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content)


def run_server(port: int = PORT):
    initialize_app()
    print("=" * 70)
    print(f"🚀 Sentix Enterprise Server active at: http://localhost:{port}")
    print("=" * 70)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), AppRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[!] Server stopped cleanly.")


if __name__ == "__main__":
    run_server()
