"""
Configuration and constants for the Product Review Sentiment Analysis System.
"""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# File paths
AMAZON_DATA_PATH = DATA_DIR / "amazon_product_reviews.csv"
DEFAULT_TRAIN_DATA = AMAZON_DATA_PATH if AMAZON_DATA_PATH.exists() else (DATA_DIR / "sample_product_reviews.csv")
DEFAULT_TEST_DATA = DATA_DIR / "test_reviews.csv"
MODEL_SAVE_PATH = MODELS_DIR / "sentiment_classifier.json"
STAR_MODEL_SAVE_PATH = MODELS_DIR / "star_rating_classifier.json"
DASHBOARD_OUTPUT_PATH = OUTPUT_DIR / "sentiment_dashboard.html"

# Aspects of interest in product reviews
ASPECT_CATEGORIES = {
    "battery": [
        "battery", "charge", "charging", "charger", "standby", "battery life",
        "power", "mah", "discharge", "plugged"
    ],
    "quality": [
        "quality", "build", "material", "durability", "durable", "sturdy",
        "fragile", "plastic", "craftsmanship", "finish", "broken", "flimsy"
    ],
    "price": [
        "price", "cost", "value", "expensive", "cheap", "worth", "affordable",
        "deal", "overpriced", "bargain", "money"
    ],
    "delivery": [
        "delivery", "shipping", "shipped", "package", "packaging", "courier",
        "arrived", "box", "transit", "delay", "received"
    ],
    "usability": [
        "easy", "difficult", "setup", "install", "installation", "manual",
        "interface", "intuitive", "user-friendly", "confusing", "controls"
    ]
}

# Standard English stopwords (to keep pure-Python execution dependency-free)
STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves"
])

# Negation words critical for sentiment polarity
NEGATION_WORDS = set([
    "not", "no", "never", "none", "neither", "nor", "hardly", "barely",
    "scarcely", "without", "rarely", "n't", "cannot"
])

# Sentiment Lexicon for Aspect-Based Sentiment Analysis & Rule engine
POSITIVE_LEXICON = set([
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "superb",
    "outstanding", "impressive", "terrific", "stellar", "love", "loved", "like",
    "liked", "best", "perfect", "flawless", "reliable", "solid", "durable",
    "fast", "responsive", "smooth", "crisp", "clear", "clearer", "sharp",
    "comfortable", "sleek", "modern", "affordable", "cheap", "bargain", "worth",
    "happy", "satisfied", "pleased", "recommend", "recommended", "superior",
    "seamless", "intuitive", "helpful", "easy", "friendly", "prompt", "secure"
])

NEGATIVE_LEXICON = set([
    "bad", "terrible", "horrible", "awful", "poor", "worse", "worst", "hate",
    "hated", "dislike", "disappointing", "disappointed", "disappointment", "broken",
    "defective", "flimsy", "cheap", "fragile", "slow", "sluggish", "laggy",
    "noisy", "uncomfortable", "ugly", "overpriced", "expensive", "waste", "useless",
    "regret", "unreliable", "trash", "garbage", "junk", "faulty", "failed",
    "fail", "failing", "stopped", "stuck", "frustrating", "annoying", "poorly",
    "clunky", "delayed", "late", "damaged", "scratched", "confusing", "difficult"
])
