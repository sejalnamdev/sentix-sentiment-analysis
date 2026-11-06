"""
Amazon Product Reviews Dataset Loader & Generator (500+ Comprehensive Reviews).

Generates a large-scale, high-fidelity Amazon Product Reviews dataset across
6 major departments with genuine Amazon schema:
[asin, product_category, product_name, review_headline, review_body,
 star_rating, helpful_votes, total_votes, verified_purchase, sentiment, primary_aspect]
"""
import csv
import random
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AMAZON_CSV_PATH = DATA_DIR / "amazon_product_reviews.csv"

# Comprehensive product catalog across 6 Amazon departments
PRODUCTS = [
    # Electronics - Audio
    ("B0863TXGM3", "Electronics", "Sony WH-1000XM4 Wireless Noise-Canceling Headphones"),
    ("B09JQMJHXY", "Electronics", "Apple AirPods Pro (2nd Generation)"),
    ("B07PXGQC1Q", "Electronics", "Anker Soundcore 2 Portable Bluetooth Speaker"),
    ("B098FKXT8L", "Electronics", "Bose QuietComfort 45 Bluetooth Headphones"),
    ("B07Q9MJKBV", "Electronics", "JBL Flip 6 Waterproof Portable Bluetooth Speaker"),

    # Computers & Accessories
    ("B08N5WRWNW", "Computers", "Apple MacBook Air 13-inch (M1 Chip 256GB)"),
    ("B09SWW583J", "Computers", "Logitech MX Master 3S Wireless Performance Mouse"),
    ("B08HGMW7M9", "Computers", "Dell UltraSharp 27-inch 4K UHD Monitor"),
    ("B08T6K9Q4B", "Computers", "Razer BlackWidow V3 Mechanical Gaming Keyboard"),
    ("B08D2JBV2W", "Computers", "Samsung 980 PRO 1TB PCIe NVMe Gen4 SSD"),

    # Smart Home & Devices
    ("B09B8V1LZ3", "Smart Home", "Amazon Echo Dot (5th Gen) Smart Speaker with Alexa"),
    ("B08KTZ8249", "Smart Home", "Kindle Paperwhite (16 GB) 6.8-inch Glare-Free Display"),
    ("B08XVVP8PH", "Smart Home", "Amazon Fire TV Stick 4K Max Streaming Device"),
    ("B08N5N6RSS", "Smart Home", "Ring Video Doorbell Wired with 1080p HD Video"),
    ("B09H26Y34K", "Smart Home", "Google Nest Learning Thermostat 3rd Generation"),

    # Home & Kitchen
    ("B07PV3X97B", "Home & Kitchen", "Instant Pot Duo 7-in-1 Electric Pressure Cooker 6 Qt"),
    ("B07FX7Z3RN", "Home & Kitchen", "Ninja AF101 Air Fryer 4 Qt Dishwasher Safe"),
    ("B08924WV7M", "Home & Kitchen", "Dyson V8 Cordless Stick Vacuum Cleaner"),
    ("B0748J568X", "Home & Kitchen", "Keurig K-Elite Single Serve Coffee Maker"),
    ("B008H4SLV6", "Home & Kitchen", "Cuisinart 14-Cup Food Processor Stainless Steel"),

    # Health & Personal Care
    ("B08K9B6X8T", "Health & Personal", "Philips Sonicare ProtectiveClean 4100 Electric Toothbrush"),
    ("B098FBHQ94", "Health & Personal", "Fitbit Charge 6 Fitness Tracker with Heart Rate"),
    ("B099KTY897", "Health & Personal", "Oral-B Pro 1000 CrossAction Electric Toothbrush"),
    ("B08J89P1KL", "Health & Personal", "Dyson Supersonic Hair Dryer Ceramic Pink"),
    ("B094R8M2WQ", "Health & Personal", "Theragun Mini Deep Tissue Massage Gun"),

    # Sports & Outdoors
    ("B01KXHH3DY", "Sports & Outdoors", "Hydro Flask Standard Mouth Water Bottle 24 oz"),
    ("B079F4FGF7", "Sports & Outdoors", "Baleaf Men's Athletic Quick Dry Running Shorts"),
    ("B083N3TY2M", "Sports & Outdoors", "Garmin Forerunner 55 GPS Running Smartwatch"),
    ("B000G64V2K", "Sports & Outdoors", "Coleman Sundome Camping Tent 4 Person"),
    ("B079D9FHYN", "Sports & Outdoors", "Yeti Rambler 20 oz Stainless Steel Tumbler")
]

# Review templates designed with authentic Amazon customer patterns
REVIEW_TEMPLATES = {
    "positive": [
        # Battery focus
        ("{product} has exceptional battery life! I get well over {hours} hours of continuous use on a single charge. The build quality feels solid and premium.", "Outstanding battery life that exceeds expectations!", "battery", 5),
        ("Fast charging is a game changer on this {product}. Takes only 30 minutes to get 70% juice. Standby power consumption is virtually zero.", "Super fast charging and long standby time", "battery", 5),
        # Quality focus
        ("Build quality is phenomenal. You can immediately feel the premium materials. It has survived several drops without a single scratch.", "Durable, high-end build and craftsmanship", "quality", 5),
        ("Flawless performance out of the box! The sound and tactile feedback are crisp and satisfying. Definitely lives up to the reputable brand name.", "Top notch quality and flawless execution", "quality", 5),
        ("Very impressed by the durability. The materials are sturdy and well engineered. I use it every single day and it still looks brand new.", "Solid craftsmanship that holds up to daily heavy use", "quality", 5),
        # Price focus
        ("Outstanding value for the price. Performs just as well as competing brands that cost three times as much. Worth every single penny.", "Incredible value for money, highly recommend!", "price", 5),
        ("Caught this on an Amazon lightning deal and it was the best purchase I have made all year. Affordable without compromising on performance.", "Fantastic bargain and worth every dollar", "price", 4),
        # Usability focus
        ("Setup took less than 2 minutes. The companion app and controls are remarkably intuitive. My elderly parents were able to use it effortlessly.", "Extremely intuitive and user-friendly setup", "usability", 5),
        ("Ergonomic and comfortable to use for hours on end. The controls are responsive and navigation is smooth as butter.", "Remarkably comfortable and effortless to operate", "usability", 5),
        # Delivery focus
        ("Shipped extremely fast via Amazon Prime. Packaging was pristine with reinforced corners. Arrived two days ahead of schedule.", "Prompt delivery and well-protected packaging", "delivery", 5)
    ],
    "negative": [
        # Battery focus
        ("The battery on this {product} drains drastically fast. Barely lasts {bad_hours} hours before requiring a recharge. Very frustrating.", "Terrible battery life, constantly dying", "battery", 1),
        ("The charger stopped working after only three weeks. When plugged into the wall, the unit overheats dangerously. Cheap power components.", "Defective charging unit and dangerous overheating", "battery", 1),
        # Quality focus
        ("The build quality is extremely disappointing. Flimsy plastic construction that feels cheap. A plastic hinge broke after light normal use.", "Cheap materials and broke within a month", "quality", 1),
        ("Stopped working completely after 45 days. The internal motor began rattling loudly and smoke came out. Zero durability.", "Complete failure after moderate use, low durability", "quality", 1),
        ("Scratched right out of the sealed box. The finish looks sloppy and quality control was clearly asleep. Will be returning.", "Poor quality control and cosmetic defects", "quality", 2),
        # Price focus
        ("Massively overpriced for what it delivers. You are purely paying for the brand logo. Generic alternatives work better at half the cost.", "Ridiculously overpriced and underwhelming", "price", 2),
        ("A total waste of money. Does not perform the basic features advertised. Regret buying this product completely.", "Complete waste of hard-earned money", "price", 1),
        # Usability focus
        ("The user interface is an absolute nightmare. Confusing buttons, poorly written instruction manual, and constant software lag.", "Confusing manual and frustrating interface", "usability", 2),
        ("Horrible ergonomic design. Causes painful wrist strain within 20 minutes of usage. Uncomfortable and awkward to handle.", "Painfully uncomfortable and poorly designed ergonomics", "usability", 1),
        # Delivery focus
        ("Package arrived damaged with a crushed outer box. The delivery courier left it out in the heavy rain. Customer support was unhelpful.", "Damaged delivery and crushed retail box", "delivery", 1)
    ],
    "neutral": [
        ("The sound and features on {product} are decent, but the battery life is mediocre at around {hours} hours. Average product for the cost.", "Decent performance, but nothing spectacular", "battery", 3),
        ("Does what it says on the box, but nothing about it feels premium. A good budget option if you keep your expectations in check.", "Acceptable budget item, gets the job done", "price", 3),
        ("Build is sturdy enough, but the buttons feel slightly mushy. Takes some getting used to. Neither love it nor hate it.", "Middle-of-the-road quality, decent overall", "quality", 3),
        ("Good hardware, but the mobile app requires creating an account and has occasional connection drops. Mixed feelings.", "Decent hardware held back by buggy software", "usability", 3)
    ]
}

# Sarcastic template pairs
SARCASTIC_REVIEWS = [
    ("Great phone if you love charging it 5 times a day and having it double as a hand warmer in winter!", "A truly wonderful piece of engineering!", "battery", 1),
    ("Love waiting 4 weeks for an empty crushed box, only for support to ask me to prove I did not receive it!", "Superb delivery experience!", "delivery", 1),
    ("Fantastic build quality! The swivel hinge snapped the exact day after the 30-day return window closed.", "Perfect timing on the product breakdown!", "quality", 1),
    ("Brilliant vacuum if your idea of cleaning is pushing dust around and vacuuming for 6 minutes before battery death.", "Marvelous machine for tiny 2-foot rooms", "battery", 1)
]


def generate_amazon_dataset(num_reviews: int = 500, output_path: Path = AMAZON_CSV_PATH) -> int:
    """Generates 500+ realistic Amazon customer reviews across categories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    reviews_list = []
    asin_counter = 1000

    # 1. First add dedicated sarcastic reviews
    for sarcasm_body, headline, aspect, stars in SARCASTIC_REVIEWS:
        prod = random.choice(PRODUCTS)
        reviews_list.append({
            "asin": prod[0],
            "product_category": prod[1],
            "product_name": prod[2],
            "review_headline": headline,
            "review_body": sarcasm_body,
            "star_rating": stars,
            "helpful_votes": random.randint(45, 95),
            "total_votes": random.randint(96, 110),
            "verified_purchase": "Y",
            "sentiment": "negative",
            "primary_aspect": aspect
        })

    # 2. Generate balanced reviews across categories
    weights = [0.55, 0.35, 0.10]  # 55% positive, 35% negative, 10% neutral/mixed (reflecting real Amazon)
    sentiments = ["positive", "negative", "neutral"]

    while len(reviews_list) < num_reviews:
        chosen_sentiment = random.choices(sentiments, weights=weights)[0]
        prod = random.choice(PRODUCTS)
        template, headline_tpl, aspect, stars = random.choice(REVIEW_TEMPLATES[chosen_sentiment])

        hours_val = random.randint(14, 40)
        bad_hours_val = random.randint(1, 3)

        body = template.format(
            product=prod[2],
            hours=hours_val,
            bad_hours=bad_hours_val
        )
        headline = headline_tpl

        helpful = random.randint(0, 35) if random.random() > 0.4 else random.randint(36, 120)
        total = helpful + random.randint(0, 6)

        reviews_list.append({
            "asin": prod[0],
            "product_category": prod[1],
            "product_name": prod[2],
            "review_headline": headline,
            "review_body": body,
            "star_rating": stars,
            "helpful_votes": helpful,
            "total_votes": total,
            "verified_purchase": "Y" if random.random() > 0.12 else "N",
            "sentiment": chosen_sentiment,
            "primary_aspect": aspect
        })

    fields = [
        "asin", "product_category", "product_name", "review_headline",
        "review_body", "star_rating", "helpful_votes", "total_votes",
        "verified_purchase", "sentiment", "primary_aspect"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviews_list)

    print(f"[Success] Generated comprehensive Amazon Dataset with {len(reviews_list)} reviews at: {output_path}")
    return len(reviews_list)


if __name__ == "__main__":
    generate_amazon_dataset(500)
