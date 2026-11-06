"""
LLM-Powered Sentiment and Review Intelligence Engine using Google Gemini.

Performs:
- Nuanced Sentiment Classification (Positive, Negative, Mixed, Neutral)
- Sarcasm & Irony Detection with confidence score
- Aspect-Based Sentiment Extraction (Battery, Price, Quality, Delivery, Usability)
- Star Rating Estimation (1 to 5) with reasoning
- Fallback heuristic analyzer when API key is not present
"""
import os
import json
import re
from typing import Dict, Any, Optional
from .aspect_analyzer import AspectSentimentAnalyzer

# Try importing official Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    try:
        import google.generativeai as genai_legacy
        HAS_GENAI_LEGACY = True
        HAS_GENAI = False
    except ImportError:
        HAS_GENAI = False
        HAS_GENAI_LEGACY = False


PROMPT_TEMPLATE = """You are an expert NLP Sentiment Analyst specializing in e-commerce customer reviews.
Analyze the following product review and return ONLY a valid JSON object matching the schema below:

REVIEW TO ANALYZE:
\"\"\"{review_text}\"\"\"

JSON SCHEMA:
{{
  "overall_sentiment": "positive | negative | mixed | neutral",
  "confidence": 0.0 to 1.0,
  "is_sarcastic": true or false,
  "sarcasm_explanation": "Explanation if sarcasm is detected, otherwise null",
  "predicted_star_rating": 1 to 5,
  "aspects": {{
    "battery": {{"mentioned": true/false, "sentiment": "positive|negative|neutral|null", "score": -1.0 to 1.0, "reason": "brief quote or explanation"}},
    "quality": {{"mentioned": true/false, "sentiment": "positive|negative|neutral|null", "score": -1.0 to 1.0, "reason": "brief quote or explanation"}},
    "price": {{"mentioned": true/false, "sentiment": "positive|negative|neutral|null", "score": -1.0 to 1.0, "reason": "brief quote or explanation"}},
    "delivery": {{"mentioned": true/false, "sentiment": "positive|negative|neutral|null", "score": -1.0 to 1.0, "reason": "brief quote or explanation"}},
    "usability": {{"mentioned": true/false, "sentiment": "positive|negative|neutral|null", "score": -1.0 to 1.0, "reason": "brief quote or explanation"}}
  }},
  "summary": "One sentence summary of customer feedback"
}}
"""


class LLMSentimentAnalyzer:
    """Performs deep sentiment and aspect analysis using Gemini LLM with offline fallback."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.client = None
        self.heuristic_fallback = AspectSentimentAnalyzer()

        if self.api_key and HAS_GENAI:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize Gemini Client: {e}")

    def analyze_review(self, review_text: str) -> Dict[str, Any]:
        """Runs Gemini LLM review analysis with automatic schema parsing and fallback."""
        if not review_text or not review_text.strip():
            return {"error": "Empty review text provided."}

        # If Gemini client is available, make API call
        if self.client:
            try:
                prompt = PROMPT_TEMPLATE.format(review_text=review_text)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                text_resp = response.text
                # Parse JSON
                return json.loads(text_resp)
            except Exception as err:
                print(f"[Info] Gemini API call returned: {err}. Falling back to rule-based engine.")

        # Fallback Heuristic & Rule Engine
        return self._heuristic_analysis(review_text)

    def _heuristic_analysis(self, review_text: str) -> Dict[str, Any]:
        """High-fidelity rule-based fallback when offline or no API key is supplied."""
        raw_res = self.heuristic_fallback.analyze_review(review_text)
        aspect_data = raw_res.get("aspects", raw_res)
        text_lower = review_text.lower()

        # Detect potential sarcasm cues
        sarcasm_detected = False
        sarcasm_reason = None
        sarcastic_patterns = [
            (r"great.*if you (like|love|enjoy|want).*(broken|peeling|waiting|tethered|terrible)", "Contrasting praise followed by ironic penalty condition."),
            (r"love.*waiting.*weeks", "Exaggerated positive sentiment used to describe severe delivery delay."),
            (r"wonderful.*stopped working", "Positive adjective directly contradictory to failure report.")
        ]
        for pattern, reason in sarcastic_patterns:
            if re.search(pattern, text_lower):
                sarcasm_detected = True
                sarcasm_reason = reason
                break

        # Compute aggregate sentiment from aspects and text cues
        pos_scores = [d["score"] for d in aspect_data.values() if d["mentioned"] and d["score"] > 0]
        neg_scores = [d["score"] for d in aspect_data.values() if d["mentioned"] and d["score"] < 0]

        total_pos = sum(pos_scores)
        total_neg = abs(sum(neg_scores))

        if sarcasm_detected:
            overall_sentiment = "negative"
            predicted_star = 1
            confidence = 0.88
        elif total_pos > 0 and total_neg > 0:
            overall_sentiment = "mixed"
            predicted_star = 3
            confidence = 0.80
        elif total_pos > total_neg:
            overall_sentiment = "positive"
            predicted_star = 5 if total_pos >= 1.5 else 4
            confidence = 0.90
        elif total_neg > total_pos:
            overall_sentiment = "negative"
            predicted_star = 1 if total_neg >= 1.5 else 2
            confidence = 0.90
        else:
            overall_sentiment = "neutral"
            predicted_star = 3
            confidence = 0.70

        aspects_formatted = {}
        for asp, data in aspect_data.items():
            aspects_formatted[asp] = {
                "mentioned": data["mentioned"],
                "sentiment": data["sentiment"] if data["mentioned"] else None,
                "score": data["score"],
                "reason": data["snippets"][0] if data["snippets"] else None
            }

        return {
            "overall_sentiment": overall_sentiment,
            "confidence": confidence,
            "is_sarcastic": sarcasm_detected,
            "sarcasm_explanation": sarcasm_reason,
            "predicted_star_rating": predicted_star,
            "aspects": aspects_formatted,
            "summary": f"Review expressing {overall_sentiment} feedback with estimated {predicted_star} star rating."
        }
