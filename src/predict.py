"""Dự báo giá nhà từ thông tin đầu vào."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from regions import region_display_name, resolve_city_for_model
from utils import (
    FURNISHED_FLOOR,
    FURNISHED_ORDINAL,
    FURNISHED_OPTIONS,
    LEGAL_FLOOR,
    LEGAL_ORDINAL,
    LEGAL_OPTIONS,
    PROPERTY_TYPES,
    apply_calibration,
    format_vnd,
    format_vnd_full,
    get_baseline_unit,
    load_model,
    normalize_property_type,
    reference_item_to_display,
)


def get_reference_examples(
    bundle: dict,
    property_type: str,
    city: str,
    area_m2: float,
    bedrooms: int,
    bathrooms: int,
    legal_status: str,
    furnished_status: str,
    predicted_price: float,
    limit: int = 3,
) -> list[dict]:
    pool = bundle.get("reference_pool", [])
    if not pool:
        return []

    # Bắt buộc cùng khu vực user chọn; không fallback sang khu vực khác.
    city_pool = [x for x in pool if x.get("city") == city]
    candidates = [x for x in city_pool if x.get("property_type") == property_type]
    if len(candidates) < limit:
        candidates = city_pool

    scored: list[tuple[float, dict]] = []
    for item in candidates:
        try:
            item_price = float(item.get("price_vnd", 0) or 0)
            item_area = float(item.get("area_m2", 0) or 0)
            item_bed = int(item.get("bedrooms", 0) or 0)
            item_bath = int(item.get("bathrooms", 0) or 0)
        except (TypeError, ValueError):
            continue
        if item_price <= 0 or item_area <= 0:
            continue

        # Giữ lại tin có thông số tương đối giống nhau.
        area_ratio = item_area / max(area_m2, 1.0)
        if area_ratio < 0.7 or area_ratio > 1.35:
            continue
        if abs(item_bed - bedrooms) > 2:
            continue
        if abs(item_bath - bathrooms) > 2:
            continue
        price_ratio = item_price / max(predicted_price, 1.0)
        if price_ratio < 0.55 or price_ratio > 1.6:
            continue

        score = 0.0
        score += abs(item_area - area_m2) / max(area_m2, 1)
        score += 0.18 * abs(item_bed - bedrooms)
        score += 0.14 * abs(item_bath - bathrooms)
        score += 0.22 * (abs(item_price - predicted_price) / max(predicted_price, 1))
        score += 0.08 * float(item.get("legal_status") != legal_status)
        score += 0.08 * float(item.get("furnished_status") != furnished_status)
        scored.append((score, item))

    scored.sort(key=lambda x: x[0])
    picked: list[dict] = []
    seen = set()
    for _, item in scored:
        url = str(item.get("source_url", "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        picked.append(reference_item_to_display(item))
        if len(picked) == limit:
            break
    return picked


def predict_price(
    property_type: str,
    city: str,
    area_m2: float,
    bedrooms: int = 1,
    bathrooms: int = 1,
    legal_status: str = "unknown",
    furnished_status: str = "unknown",
    frontage_m: float = 0,
    model_path: Path | None = None,
) -> dict:
    bundle = load_model(model_path)
    pipeline = bundle["pipeline"]
    calibration = bundle.get("calibration", {})
    baselines = bundle.get("baselines", {})

    property_type = normalize_property_type(property_type)

    city = resolve_city_for_model(city, bundle["city_options"])

    legal_status = legal_status.strip() or "unknown"
    furnished_status = furnished_status.strip() or "unknown"
    if legal_status not in LEGAL_OPTIONS:
        raise ValueError("Pháp lý không hợp lệ.")
    if furnished_status not in FURNISHED_OPTIONS:
        raise ValueError("Tình trạng nội thất không hợp lệ.")

    if area_m2 <= 0:
        raise ValueError("Diện tích phải lớn hơn 0.")

    bedrooms = max(0, min(int(bedrooms), 6))
    bathrooms = max(1, min(int(bathrooms), 4))
    frontage_m = max(0.0, float(frontage_m or 0))

    if property_type == "can_ho_chung_cu" and bedrooms < 1:
        raise ValueError("Căn hộ cần ít nhất 1 phòng ngủ.")

    sample = pd.DataFrame(
        [
            {
                "property_type": property_type,
                "city": city,
                "area_m2": float(area_m2),
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "frontage_m": frontage_m,
                "road_width_m": 0.0,
                "total_floors": 0.0,
                "legal_ordinal": float(LEGAL_ORDINAL["hop_dong"]),
                "furnished_ordinal": float(FURNISHED_ORDINAL["co_ban"]),
                "room_density": (bedrooms + bathrooms * 0.5) / max(area_m2, 15),
                "livability_score": 0.5,
            }
        ]
    )

    unit_ml = float(pipeline.predict(sample)[0])
    blend_weight = float(bundle.get("blend_weight", 1.0))
    unit_base = get_baseline_unit(property_type, city, baselines)
    unit_price = blend_weight * unit_ml + (1 - blend_weight) * unit_base
    predicted_price = unit_price * area_m2

    # Giá sàn theo cấu hình tham chiếu 2PN + hợp đồng (tránh ML đảo ngược pháp lý/phòng)
    anchor_sample = pd.DataFrame(
        [
            {
                "property_type": property_type,
                "city": city,
                "area_m2": float(area_m2),
                "bedrooms": 2,
                "bathrooms": max(1, min(bathrooms, 2)),
                "frontage_m": 0.0,
                "road_width_m": 0.0,
                "total_floors": 0.0,
                "legal_ordinal": float(LEGAL_ORDINAL["hop_dong"]),
                "furnished_ordinal": float(FURNISHED_ORDINAL["co_ban"]),
                "room_density": (2 + min(bathrooms, 2) * 0.5) / max(area_m2, 15),
                "livability_score": 0.5,
            }
        ]
    )
    anchor_unit_ml = float(pipeline.predict(anchor_sample)[0])
    anchor_unit = blend_weight * anchor_unit_ml + (1 - blend_weight) * unit_base
    anchor_price = anchor_unit * area_m2
    room_step = calibration.get("bedroom_step", {}).get(property_type, 0.04)
    floor_mult = LEGAL_FLOOR.get(legal_status, 1.0) * FURNISHED_FLOOR.get(
        furnished_status, 1.0
    ) * (
        (1 + room_step) ** max(0, bedrooms - 2)
    )
    if property_type in bundle.get("frontage_types", []) and frontage_m > 4:
        fstep = calibration.get("frontage_step", {}).get(property_type, 0.01)
        floor_mult *= 1 + fstep * (frontage_m - 4)
    predicted_price = max(predicted_price, anchor_price * floor_mult)

    predicted_price = apply_calibration(
        predicted_price,
        property_type,
        legal_status,
        furnished_status,
        bedrooms,
        frontage_m,
        calibration,
    )

    references = get_reference_examples(
        bundle=bundle,
        property_type=property_type,
        city=city,
        area_m2=area_m2,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        legal_status=legal_status,
        furnished_status=furnished_status,
        predicted_price=predicted_price,
        limit=10,
    )

    legal_labels = bundle.get("legal_options", LEGAL_OPTIONS)
    furnished_labels = bundle.get("furnished_options", FURNISHED_OPTIONS)

    return {
        "predicted_price": predicted_price,
        "predicted_price_short": format_vnd(predicted_price),
        "predicted_price_full": format_vnd_full(predicted_price),
        "unit_price_m2": predicted_price / area_m2,
        "unit_price_short": format_vnd(predicted_price / area_m2),
        "references": references,
        "inputs": {
            "property_type": PROPERTY_TYPES[property_type],
            "city": region_display_name(city),
            "city_code": city,
            "area_m2": area_m2,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "legal_status": legal_labels.get(legal_status, legal_status),
            "furnished_status": furnished_labels.get(
                furnished_status, furnished_status
            ),
            "frontage_m": frontage_m if frontage_m > 0 else None,
        },
    }


if __name__ == "__main__":
    low = predict_price(
        property_type="can_ho_chung_cu",
        city="tp_ho_chi_minh",
        area_m2=80,
        bedrooms=2,
        bathrooms=2,
        legal_status="hop_dong",
        furnished_status="co_ban",
    )
    high = predict_price(
        property_type="nha_mat_pho",
        city="tp_ho_chi_minh",
        area_m2=80,
        bedrooms=4,
        bathrooms=3,
        legal_status="so_do",
        furnished_status="co_ban",
        frontage_m=8,
    )
    print("Căn hộ HĐMB 2PN:", low["predicted_price_short"])
    print("Mặt phố sổ đỏ 4PN 8m MT:", high["predicted_price_short"])
