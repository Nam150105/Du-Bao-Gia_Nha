"""Ứng dụng web dự báo giá nhà."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

from flask import Flask, Response, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import predict_price

from regions import REGION_GROUPS, REGION_OPTIONS
from utils import (
    DEFAULT_CITY,
    FRONTAGE_TYPES,
    FURNISHED_OPTIONS,
    LEGAL_OPTIONS,
    MODEL_PATH,
    PROPERTY_TYPES,
    get_map_points,
    get_random_references,
    load_model,
    proxy_listing_image_bytes,
)

app = Flask(__name__)


def get_model_info() -> dict:
    try:
        bundle = load_model()
        return {
            "ready": True,
            "metrics": bundle.get("metrics", {}),
        }
    except FileNotFoundError:
        return {"ready": False, "metrics": {}}


def get_initial_references(limit: int = 24, city: str = DEFAULT_CITY) -> list[dict]:
    try:
        bundle = load_model()
    except FileNotFoundError:
        return []
    pool = bundle.get("reference_pool", [])
    city_pool = [x for x in pool if x.get("city") == city]
    if city_pool:
        pool = city_pool
    return get_random_references(pool, limit=limit)


@app.route("/")
def index():
    model_info = get_model_info()
    return render_template(
        "index.html",
        property_types=PROPERTY_TYPES,
        frontage_types=list(FRONTAGE_TYPES),
        region_groups=REGION_GROUPS,
        region_options=REGION_OPTIONS,
        legal_options=LEGAL_OPTIONS,
        furnished_options=FURNISHED_OPTIONS,
        model_ready=model_info["ready"],
        metrics=model_info["metrics"],
        initial_references=get_initial_references(limit=24, city=DEFAULT_CITY),
        default_city=DEFAULT_CITY,
        default_city_label=REGION_OPTIONS.get(DEFAULT_CITY, "Hà Nội"),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form
        result = predict_price(
            property_type=data.get("property_type", "").strip(),
            city=data.get("city", "").strip(),
            area_m2=float(data.get("area_m2", 0)),
            bedrooms=int(data.get("bedrooms", 1) or 1),
            bathrooms=int(data.get("bathrooms", 1) or 1),
            legal_status=data.get("legal_status", "unknown").strip(),
            furnished_status=data.get("furnished_status", "unknown").strip(),
            frontage_m=float(data.get("frontage_m", 0) or 0),
        )
        return jsonify({"success": True, "result": result})
    except (ValueError, TypeError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503


@app.route("/api/listing-image")
def listing_image_legacy():
    """Tương thích JS cũ: trả URL proxy (trình duyệt cache có thể vẫn gọi endpoint này)."""
    listing_url = request.args.get("url", "").strip()
    if not listing_url.startswith("http"):
        return jsonify({"success": False, "image_url": None}), 400
    proxy_url = f"/api/image-proxy?url={quote(listing_url, safe='')}"
    return jsonify({"success": True, "image_url": proxy_url})


@app.route("/api/image-proxy")
def image_proxy():
    listing_url = request.args.get("url", "").strip()
    if not listing_url.startswith("http"):
        return Response("Invalid URL", status=400)
    result = proxy_listing_image_bytes(listing_url)
    if not result:
        return Response(status=404)
    data, content_type = result
    return Response(
        data,
        mimetype=content_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.route("/api/map/points")
def map_points():
    city = request.args.get("city", DEFAULT_CITY).strip()
    property_type = request.args.get("property_type", "").strip() or None
    show_all = request.args.get("all", "").strip().lower() in {"1", "true", "yes"}
    try:
        limit = int(request.args.get("limit", 400))
    except (TypeError, ValueError):
        limit = 400
    highlight_raw = request.args.get("highlight_urls", "")
    highlight_urls = [u.strip() for u in highlight_raw.split(",") if u.strip()]
    try:
        payload = get_map_points(
            city=city,
            property_type=property_type,
            limit=limit,
            show_all=show_all,
            highlight_urls=highlight_urls or None,
        )
        return jsonify({"success": True, "map": payload})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_exists": MODEL_PATH.exists()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
