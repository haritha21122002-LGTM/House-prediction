"""Flask API and web server for Bengaluru house price predictions."""

import csv
import pickle
from pathlib import Path

from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "Bengaluru_House_Data_Cleaned.csv"
FEATURES = [
    "area_type",
    "availability",
    "location",
    "total_sqft",
    "bhk",
    "bath",
    "balcony",
]

app = Flask(__name__)


def load_options():
    options = {key: set() for key in ("area_type", "availability", "location")}
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            for key in options:
                value = (row.get(key) or "").strip()
                if value:
                    options[key].add(value)
    return {key: sorted(values) for key, values in options.items()}


def load_analytics():
    groups = {"bhk": {}, "area_type": {}, "location": {}}
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            try:
                price = float(row["price"])
                bhk = str(int(float(row["bhk"])))
            except (KeyError, TypeError, ValueError):
                continue
            for key, label in (("bhk", bhk), ("area_type", row.get("area_type", "").strip()),
                               ("location", row.get("location", "").strip())):
                if label:
                    total, count = groups[key].get(label, (0.0, 0))
                    groups[key][label] = (total + price, count + 1)

    def averages(group):
        return [{"label": label, "average_price": round(total / count, 2), "listings": count}
                for label, (total, count) in group.items()]

    by_bhk = sorted(averages(groups["bhk"]), key=lambda item: int(item["label"]))
    by_area = sorted(averages(groups["area_type"]), key=lambda item: item["average_price"], reverse=True)
    top_locations = sorted(
        (item for item in averages(groups["location"]) if item["listings"] >= 10),
        key=lambda item: item["average_price"], reverse=True
    )[:8]
    return {"by_bhk": by_bhk, "by_area_type": by_area, "top_locations": top_locations}


OPTIONS = load_options()
ANALYTICS = load_analytics()
with MODEL_PATH.open("rb") as source:
    MODEL = pickle.load(source)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/options")
def get_options():
    return jsonify(OPTIONS)


@app.get("/api/analytics")
def get_analytics():
    return jsonify(ANALYTICS)


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        row = {
            "area_type": str(payload["area_type"]).strip(),
            "availability": str(payload["availability"]).strip(),
            "location": str(payload["location"]).strip(),
            "total_sqft": float(payload["total_sqft"]),
            "bhk": float(payload["bhk"]),
            "bath": float(payload["bath"]),
            "balcony": float(payload["balcony"]),
        }
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Enter all house details using valid numbers."}), 400

    if any(not row[key] for key in ("area_type", "availability", "location")):
        return jsonify({"error": "Choose an area type, availability, and location."}), 400
    if any(row[key] < 0 for key in ("total_sqft", "bhk", "bath", "balcony")):
        return jsonify({"error": "Numeric values cannot be negative."}), 400

    import pandas as pd

    house = pd.DataFrame([row], columns=FEATURES)
    price_lakh = float(MODEL.predict(house)[0])
    return jsonify({"price_lakh": round(price_lakh, 2), "price_rupees": round(price_lakh * 100000)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
