"""Tiện ích xử lý dữ liệu và huấn luyện mô hình dự báo giá nhà."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "house_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.pkl"
DEFAULT_CITY = "ha_noi"

PROPERTY_TYPES: dict[str, str] = {
    "can_ho_chung_cu": "Căn hộ chung cư",
    "nha_rieng": "Nhà riêng",
    "biet_thu": "Nhà biệt thự",
    "lien_ke": "Liền kề",
    "nha_mat_pho": "Nhà mặt phố",
}

FRONTAGE_TYPES = {"nha_rieng", "biet_thu", "lien_ke", "nha_mat_pho"}

EXCLUDE_ASSET_TYPES = {"Đất nền", "Văn phòng", "Đất"}

LAND_TITLE_PATTERN = re.compile(
    r"đất nền|bán đất|đất thổ cư|lô đất|mặt bằng|đất trống|"
    r"đất vườn|đất trồng|đất ở ngoài|đất dự án",
    re.IGNORECASE,
)

from regions import (
    REGION_GROUPS,
    REGION_OPTIONS,
    extract_region_code,
    region_display_name,
)

# Giữ tên cũ cho tương thích; gồm 34 tỉnh/thành + mã khac khi không map được
CITY_OPTIONS: dict[str, str] = {
    **REGION_OPTIONS,
    "khac": "Khác / chưa xác định",
}

LEGAL_OPTIONS = {
    "so_do": "Sổ đỏ / Sổ hồng",
    "hop_dong": "Hợp đồng mua bán",
    "cho_so": "Đang chờ sổ",
    "unknown": "Không rõ",
}

LEGAL_ORDINAL = {"unknown": 0, "hop_dong": 1, "cho_so": 1, "so_do": 2}

LEGAL_FLOOR = {
    "so_do": 1.06,
    "hop_dong": 1.0,
    "cho_so": 0.97,
    "unknown": 0.94,
}

FURNISHED_OPTIONS = {
    "full": "Đầy đủ nội thất",
    "co_ban": "Nội thất cơ bản",
    "khong": "Không nội thất",
    "unknown": "Không rõ",
}

FURNISHED_ORDINAL = {"unknown": 1, "khong": 0, "co_ban": 1, "full": 2}
FURNISHED_FLOOR = {
    "full": 1.04,
    "co_ban": 1.0,
    "unknown": 0.96,
    "khong": 0.93,
}

CAT_FEATURES = ["property_type", "city"]
NUM_FEATURES = [
    "area_m2",
    "bedrooms",
    "bathrooms",
    "frontage_m",
    "road_width_m",
    "total_floors",
    "legal_ordinal",
    "furnished_ordinal",
    "room_density",
    "livability_score",
]


def extract_city(province: str | float, address: str | float = "") -> str:
    return extract_region_code(province, address)


def is_land_listing(row: pd.Series) -> bool:
    asset = str(row.get("asset_type", "")).strip()
    if asset in EXCLUDE_ASSET_TYPES:
        return True
    title = str(row.get("title", ""))
    if LAND_TITLE_PATTERN.search(title):
        return True
    return False


def infer_property_type(row: pd.Series) -> str:
    asset = str(row.get("asset_type", "")).strip()
    row_type = str(row.get("type", "")).strip()
    title = str(row.get("title", "")).lower()

    if row_type == "apartment" or asset == "Chung cư":
        return "can_ho_chung_cu"
    if asset == "Biệt thự" or "biệt thự" in title or "villa" in title:
        return "biet_thu"
    if "liền kề" in title or "townhouse" in title:
        return "lien_ke"
    if "mặt phố" in title or "nhà phố" in title or "shophouse" in title:
        return "nha_mat_pho"
    return "nha_rieng"


def parse_legal_status(row: pd.Series) -> str:
    title = str(row.get("title", "")).lower()
    legal_text = str(row.get("legal", "")).lower()
    legal_clean = row.get("legal_clean")

    if legal_clean is True or str(legal_clean).lower() in ("true", "1", "yes"):
        return "so_do"
    if row.get("has_certificate") is True:
        return "so_do"
    if re.search(r"sổ hồng|sổ đỏ|shr|sổ riêng|pink book", title, re.I):
        return "so_do"
    if "hợp đồng" in legal_text:
        return "hop_dong"
    if "chờ sổ" in legal_text or "đang chờ" in legal_text:
        return "cho_so"
    if re.search(r"hợp đồng", title):
        return "hop_dong"
    return "unknown"


def impute_bedrooms(row: pd.Series) -> int:
    bedrooms = row.get("bedrooms")
    if pd.notna(bedrooms) and bedrooms > 0:
        return int(min(bedrooms, 6))

    title = str(row.get("title", ""))
    match = re.search(r"(\d+)\s*pn", title, re.I)
    if match:
        return int(min(int(match.group(1)), 6))

    property_type = row["property_type"]
    area_m2 = float(row["area_m2"])
    if property_type == "can_ho_chung_cu":
        if area_m2 < 45:
            return 1
        if area_m2 < 65:
            return 2
        if area_m2 < 90:
            return 3
        return 4
    if property_type == "biet_thu":
        return int(np.clip(round(area_m2 / 80), 3, 6))
    if property_type in {"nha_mat_pho", "lien_ke"}:
        return int(np.clip(round(area_m2 / 45), 2, 5))
    return int(np.clip(round(area_m2 / 40), 2, 5))


def parse_furnished_status(row: pd.Series) -> str:
    furniture = str(row.get("furniture", "")).lower()
    if not furniture or furniture == "nan":
        return "unknown"
    if any(x in furniture for x in ("đầy đủ", "có nội thất", "full")):
        return "full"
    if any(x in furniture for x in ("cơ bản", "đang ở")):
        return "co_ban"
    if any(x in furniture for x in ("trống", "không nội thất", "chưa sử dụng")):
        return "khong"
    return "unknown"


def normalize_property_type(value: str) -> str:
    aliases = {
        "Chung cư": "can_ho_chung_cu",
        "Căn hộ": "can_ho_chung_cu",
        "Căn hộ chung cư": "can_ho_chung_cu",
        "Nhà riêng": "nha_rieng",
        "Nhà biệt thự": "biet_thu",
        "Liền kề": "lien_ke",
        "Nhà mặt phố": "nha_mat_pho",
    }
    code = aliases.get(value.strip(), value.strip())
    if code not in PROPERTY_TYPES:
        raise ValueError(
            f"Loại nhà không hợp lệ: {value}. "
            "Hệ thống chỉ dự báo nhà ở, không dự báo đất."
        )
    return code


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_PATH
    return pd.read_csv(path, low_memory=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data[~data.apply(is_land_listing, axis=1)]

    data["property_type"] = data.apply(infer_property_type, axis=1)
    data["city"] = data.apply(
        lambda r: extract_city(r.get("province"), r.get("address")), axis=1
    )

    for col in (
        "bedrooms",
        "bathrooms",
        "area_m2",
        "usable_area_m2",
        "frontage_m",
        "road_width_m",
        "total_floors",
        "price_vnd",
        "livability_score",
    ):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["legal_status"] = data.apply(parse_legal_status, axis=1)
    data["furnished_status"] = data.apply(parse_furnished_status, axis=1)

    data = data.dropna(subset=["price_vnd", "area_m2"])
    data = data[(data["price_vnd"] >= 300_000_000) & (data["area_m2"] >= 15)]

    data["unit_price_m2"] = data["price_vnd"] / data["area_m2"]
    price_cap = data["price_vnd"].quantile(0.995)
    unit_cap = data["unit_price_m2"].quantile(0.995)
    unit_floor = data["unit_price_m2"].quantile(0.01)
    data = data[
        (data["price_vnd"] <= price_cap)
        & (data["unit_price_m2"] >= max(3_000_000, unit_floor))
        & (data["unit_price_m2"] <= unit_cap)
    ]

    data["bedrooms"] = data.apply(impute_bedrooms, axis=1)
    data["bathrooms"] = data["bathrooms"].fillna(
        data["bedrooms"].apply(lambda b: max(1, min(int(b), 3)))
    ).astype(int).clip(1, 5)
    data["frontage_m"] = data["frontage_m"].fillna(0).clip(0, 50)
    data["road_width_m"] = data.get("road_width_m", pd.Series(0, index=data.index))
    data["road_width_m"] = data["road_width_m"].fillna(0).clip(0, 50)
    data["total_floors"] = data.get("total_floors", pd.Series(0, index=data.index))
    data["total_floors"] = data["total_floors"].fillna(0).clip(0, 50)
    data["livability_score"] = data.get(
        "livability_score", pd.Series(0.5, index=data.index)
    )
    data["livability_score"] = data["livability_score"].fillna(0.5).clip(0, 1)

    for col, allowed in (
        ("legal_status", set(LEGAL_OPTIONS)),
        ("furnished_status", set(FURNISHED_OPTIONS)),
    ):
        data[col] = data[col].where(data[col].isin(allowed), "unknown")

    data["legal_ordinal"] = data["legal_status"].map(LEGAL_ORDINAL).astype(float)
    data["furnished_ordinal"] = (
        data["furnished_status"].map(FURNISHED_ORDINAL).astype(float)
    )
    data["room_density"] = (data["bedrooms"] + data["bathrooms"] * 0.5) / data[
        "area_m2"
    ].clip(lower=15)

    return data.reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[CAT_FEATURES + NUM_FEATURES].copy()


def build_baselines(df: pd.DataFrame) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for (ptype, city), group in df.groupby(["property_type", "city"]):
        if len(group) < 20:
            continue
        baselines[f"{ptype}|{city}"] = float(
            (group["price_vnd"] / group["area_m2"]).median()
        )
    baselines["_global"] = float((df["price_vnd"] / df["area_m2"]).median())

    for city in df["city"].unique():
        nr_key = f"nha_rieng|{city}"
        nr_unit = baselines.get(nr_key)
        if not nr_unit:
            continue
        for ptype, factor in (
            ("nha_mat_pho", 1.15),
            ("biet_thu", 1.20),
            ("lien_ke", 1.10),
        ):
            key = f"{ptype}|{city}"
            baselines[key] = max(baselines.get(key, 0), nr_unit * factor)

    return baselines


def get_baseline_unit(property_type: str, city: str, baselines: dict) -> float:
    key = f"{property_type}|{city}"
    return baselines.get(key, baselines.get("_global", 50_000_000))


def build_calibration(df: pd.DataFrame) -> dict:
    cal: dict = {
        "legal": {},
        "bedroom_step": {},
        "frontage_step": {},
        "furnished": {},
    }
    min_samples = 30

    for ptype in df["property_type"].unique():
        sub = df[df["property_type"] == ptype]
        units = sub["unit_price_m2"]

        ref = sub[sub["legal_status"] == "hop_dong"]
        ref_unit = (
            (ref["price_vnd"] / ref["area_m2"]).median()
            if len(ref) >= min_samples
            else units.median()
        )
        for leg in LEGAL_OPTIONS:
            bucket = sub[sub["legal_status"] == leg]
            if len(bucket) < min_samples or not ref_unit:
                cal["legal"][f"{ptype}|{leg}"] = LEGAL_FLOOR.get(leg, 1.0)
                continue
            ratio = (bucket["price_vnd"] / bucket["area_m2"]).median() / ref_unit
            cal["legal"][f"{ptype}|{leg}"] = float(
                max(np.clip(ratio, 0.90, 1.28), LEGAL_FLOOR.get(leg, 1.0))
            )

        furnished_ref = sub[sub["furnished_status"] == "co_ban"]
        furnished_ref_unit = (
            (furnished_ref["price_vnd"] / furnished_ref["area_m2"]).median()
            if len(furnished_ref) >= min_samples
            else units.median()
        )
        for furn in FURNISHED_OPTIONS:
            bucket = sub[sub["furnished_status"] == furn]
            if len(bucket) < min_samples or not furnished_ref_unit:
                cal["furnished"][f"{ptype}|{furn}"] = FURNISHED_FLOOR.get(furn, 1.0)
                continue
            ratio = (bucket["price_vnd"] / bucket["area_m2"]).median() / furnished_ref_unit
            cal["furnished"][f"{ptype}|{furn}"] = float(
                max(np.clip(ratio, 0.88, 1.20), FURNISHED_FLOOR.get(furn, 1.0))
            )

        two_br = sub[sub["bedrooms"] == 2]
        base_unit = (
            (two_br["price_vnd"] / two_br["area_m2"]).median()
            if len(two_br) >= min_samples
            else units.median()
        )
        steps = []
        for br in (3, 4, 5):
            bucket = sub[sub["bedrooms"] == br]
            if len(bucket) < min_samples or not base_unit:
                continue
            ratio = (bucket["price_vnd"] / bucket["area_m2"]).median() / base_unit
            steps.append((ratio - 1.0) / (br - 2))
        cal["bedroom_step"][ptype] = float(
            np.clip(np.median(steps) if steps else 0.04, 0.02, 0.12)
        )

        if ptype in FRONTAGE_TYPES:
            with_front = sub[sub["frontage_m"] >= 4]
            no_front = sub[sub["frontage_m"] < 1]
            if len(with_front) >= min_samples and len(no_front) >= min_samples:
                u1 = (with_front["price_vnd"] / with_front["area_m2"]).median()
                u0 = (no_front["price_vnd"] / no_front["area_m2"]).median()
                cal["frontage_step"][ptype] = float(
                    np.clip((u1 / u0 - 1) / 4, 0.005, 0.04) if u0 > 0 else 0.01
                )
            else:
                cal["frontage_step"][ptype] = 0.01
        else:
            cal["frontage_step"][ptype] = 0.0

    return cal


def apply_calibration(
    price: float,
    property_type: str,
    legal_status: str,
    furnished_status: str,
    bedrooms: int,
    frontage_m: float,
    calibration: dict,
) -> float:
    price *= LEGAL_FLOOR.get(legal_status, 1.0)
    price *= FURNISHED_FLOOR.get(furnished_status, 1.0)

    extra_rooms = max(0, bedrooms - 2)
    step = calibration.get("bedroom_step", {}).get(property_type, 0.04)
    price *= (1 + step) ** extra_rooms

    if property_type in FRONTAGE_TYPES and frontage_m > 4:
        fstep = calibration.get("frontage_step", {}).get(property_type, 0.01)
        price *= 1 + fstep * (frontage_m - 4)

    return price


def extract_preview_image_from_html(html: str, page_url: str) -> str | None:
    patterns = (
        r'<meta[^>]+property=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image',
        r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image',
        r'<meta[^>]+itemprop=["\']image["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+itemprop=["\']image',
    )
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            if raw and not raw.startswith("data:"):
                return urljoin(page_url, raw)

    img_match = re.search(
        r'<img[^>]+(?:data-src|data-lazy-src|src)=["\']([^"\']+\.(?:jpg|jpeg|png|webp)[^"\']*)',
        html,
        flags=re.IGNORECASE,
    )
    if img_match:
        return urljoin(page_url, img_match.group(1).strip())
    return None


@lru_cache(maxsize=512)
def fetch_listing_preview_image(url: str) -> str | None:
    if not url.startswith("http"):
        return None
    try:
        req = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urlopen(req, timeout=4.0) as resp:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None
            html = resp.read(180_000).decode("utf-8", errors="ignore")
    except Exception:
        return None
    return extract_preview_image_from_html(html, url)


def proxy_listing_image_bytes(listing_url: str, max_bytes: int = 400_000) -> tuple[bytes, str] | None:
    """Tải ảnh từ trang tin và trả bytes để proxy qua Flask (tránh chặn hotlink)."""
    image_url = fetch_listing_preview_image(listing_url)
    if not image_url:
        return None
    try:
        req = Request(
            image_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": listing_url,
            },
        )
        with urlopen(req, timeout=5.0) as resp:
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            if not content_type.lower().startswith("image/"):
                return None
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                return None
            return data, content_type.split(";")[0].strip()
    except Exception:
        return None


def reference_item_to_display(item: dict) -> dict:
    url = str(item.get("source_url", item.get("url", ""))).strip()
    price = float(item.get("price_vnd", 0) or 0)
    cached_image = item.get("image_url")
    image_url = cached_image if cached_image else None
    lat = item.get("lat")
    lng = item.get("lng")
    try:
        lat = float(lat) if lat is not None and pd.notna(lat) else None
        lng = float(lng) if lng is not None and pd.notna(lng) else None
    except (TypeError, ValueError):
        lat, lng = None, None
    return {
        "title": str(item.get("title", "Bất động sản tham khảo")).strip()[:140],
        "url": url,
        "price_short": format_vnd(price) if price > 0 else "—",
        "area_m2": float(item.get("area_m2", 0) or 0),
        "city": region_display_name(str(item.get("city", ""))),
        "domain": urlparse(url).netloc.replace("www.", "") if url else "",
        "image_url": image_url,
        "lat": lat,
        "lng": lng,
    }


def get_random_references(pool: list[dict], limit: int = 24) -> list[dict]:
    if not pool:
        return []
    import random

    valid = [
        x
        for x in pool
        if str(x.get("source_url", "")).strip().startswith("http")
    ]
    if not valid:
        return []
    sample = random.sample(valid, min(limit, len(valid)))
    return [reference_item_to_display(x) for x in sample]


def build_reference_pool(df: pd.DataFrame, max_per_group: int = 220) -> list[dict]:
    cols = [
        "property_type",
        "city",
        "title",
        "source_url",
        "price_vnd",
        "area_m2",
        "bedrooms",
        "bathrooms",
        "legal_status",
        "furnished_status",
        "lat",
        "lng",
    ]
    cols = [c for c in cols if c in df.columns]
    ref = df[cols].copy()
    ref["source_url"] = ref["source_url"].astype(str).str.strip()
    ref = ref[ref["source_url"].str.startswith("http")]
    if ref.empty:
        return []

    samples = []
    for _, group in ref.groupby(["property_type", "city"], sort=False):
        if len(group) > max_per_group:
            group = group.sample(max_per_group, random_state=42)
        samples.append(group)
    out = pd.concat(samples, ignore_index=True) if samples else ref
    out = out.drop_duplicates(subset=["source_url"])
    return out.to_dict("records")


def _one_hot_encoder() -> OneHotEncoder:
    """HistGradientBoosting cần ma trận dày; sklearn mới mặc định sparse."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline() -> TransformedTargetRegressor:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                _one_hot_encoder(),
                CAT_FEATURES,
            ),
            ("num", "passthrough", NUM_FEATURES),
        ]
    )
    regressor = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=800,
                    max_depth=12,
                    learning_rate=0.04,
                    min_samples_leaf=30,
                    l2_regularization=0.3,
                    random_state=42,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def train_model(
    data_path: Path | None = None,
    model_path: Path | None = None,
) -> dict:
    model_path = model_path or MODEL_PATH
    df = clean_data(load_raw_data(data_path))
    features = prepare_features(df)
    target = df["unit_price_m2"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    train_df = df.loc[x_train.index]
    calibration = build_calibration(train_df)
    baselines = build_baselines(train_df)
    reference_pool = build_reference_pool(train_df)
    blend_weight = 1.0

    unit_pred = pipeline.predict(x_test)
    area_test = df.loc[x_test.index, "area_m2"].values
    predicted_prices = unit_pred * area_test

    y_true = df.loc[x_test.index, "price_vnd"]
    metrics = {
        "samples": len(df),
        "mae": float(mean_absolute_error(y_true, predicted_prices)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predicted_prices))),
        "r2": float(r2_score(y_true, predicted_prices)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "calibration": calibration,
            "baselines": baselines,
            "reference_pool": reference_pool,
            "blend_weight": blend_weight,
            "metrics": metrics,
            "property_types": PROPERTY_TYPES,
            "city_options": CITY_OPTIONS,
            "legal_options": LEGAL_OPTIONS,
            "furnished_options": FURNISHED_OPTIONS,
            "frontage_types": list(FRONTAGE_TYPES),
            "feature_columns": CAT_FEATURES + NUM_FEATURES,
        },
        model_path,
    )
    return metrics


GEO_INDEX_COLUMNS = [
    "city",
    "property_type",
    "lat",
    "lng",
    "title",
    "price_vnd",
    "area_m2",
    "bedrooms",
    "source_url",
]

CITY_MAP_CENTERS: dict[str, tuple[float, float]] = {
    "ha_noi": (21.0285, 105.8542),
    "tp_ho_chi_minh": (10.7769, 106.7009),
    "hai_phong": (20.8449, 106.6881),
    "hue": (16.4637, 107.5909),
    "da_nang": (16.0544, 108.2022),
    "can_tho": (10.0452, 105.7469),
    "dong_nai": (10.9574, 106.8426),
    "binh_duong": (11.3254, 106.4774),
    "khanh_hoa": (12.2388, 109.1967),
    "lam_dong": (11.9404, 108.4583),
    "quang_ninh": (21.0064, 107.2925),
    "nghe_an": (18.6796, 105.6813),
    "thanh_hoa": (19.8067, 105.7852),
    "bac_ninh": (21.1861, 106.0763),
}


@lru_cache(maxsize=1)
def _get_geo_index() -> pd.DataFrame:
    """Chỉ mục tọa độ nhẹ từ CSV — cache trong bộ nhớ sau lần đọc đầu."""
    raw = load_raw_data()
    data = raw[~raw.apply(is_land_listing, axis=1)].copy()
    data["city"] = data.apply(
        lambda r: extract_city(r.get("province"), r.get("address")), axis=1
    )
    data["property_type"] = data.apply(infer_property_type, axis=1)

    for col in ("lat", "lng", "price_vnd", "area_m2", "bedrooms"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["lat", "lng", "price_vnd", "area_m2"])
    data = data[
        (data["lat"].between(8.0, 24.0))
        & (data["lng"].between(102.0, 110.5))
        & (data["price_vnd"] >= 300_000_000)
        & (data["area_m2"] >= 15)
    ]

    keep = [c for c in GEO_INDEX_COLUMNS if c in data.columns]
    return data[keep].reset_index(drop=True)


def get_map_points(
    city: str,
    property_type: str | None = None,
    limit: int = 400,
    show_all: bool = False,
    highlight_urls: list[str] | None = None,
) -> dict:
    """Lấy điểm bản đồ theo khu vực; có thể trả toàn bộ hoặc mẫu giới hạn."""
    from regions import resolve_city_for_model

    city = resolve_city_for_model(city, CITY_OPTIONS)
    if not show_all:
        limit = max(10, min(int(limit), 800))
    df = _get_geo_index()
    subset = df[df["city"] == city]
    if property_type:
        try:
            ptype = normalize_property_type(property_type)
            typed = subset[subset["property_type"] == ptype]
            if len(typed) >= 20:
                subset = typed
        except ValueError:
            pass

    highlight_set = {
        u.strip()
        for u in (highlight_urls or [])
        if isinstance(u, str) and u.strip().startswith("http")
    }

    if show_all:
        picked = subset
    else:
        highlighted = pd.DataFrame()
        if highlight_set and "source_url" in subset.columns:
            highlighted = subset[subset["source_url"].astype(str).isin(highlight_set)]

        remaining = subset
        if not highlighted.empty:
            remaining = subset[~subset["source_url"].astype(str).isin(highlight_set)]

        sample_size = max(0, limit - len(highlighted))
        if len(remaining) > sample_size:
            remaining = remaining.sample(sample_size, random_state=42)

        picked = pd.concat([highlighted, remaining], ignore_index=True)
    if picked.empty:
        center = CITY_MAP_CENTERS.get(city, (16.0, 107.0))
        return {
            "city": city,
            "city_label": region_display_name(city),
            "total_in_city": 0,
            "shown": 0,
            "center": {"lat": center[0], "lng": center[1]},
            "bounds": None,
            "points": [],
        }

    points: list[dict] = []
    for row in picked.itertuples(index=False):
        url = str(getattr(row, "source_url", "") or "").strip()
        price = float(getattr(row, "price_vnd", 0) or 0)
        points.append(
            {
                "lat": float(row.lat),
                "lng": float(row.lng),
                "title": str(getattr(row, "title", "")).strip()[:120],
                "price_short": format_vnd(price) if price > 0 else "—",
                "area_m2": float(getattr(row, "area_m2", 0) or 0),
                "url": url,
                "highlight": url in highlight_set,
            }
        )

    lats = [p["lat"] for p in points]
    lngs = [p["lng"] for p in points]
    center = CITY_MAP_CENTERS.get(
        city,
        (float(sum(lats) / len(lats)), float(sum(lngs) / len(lngs))),
    )

    return {
        "city": city,
        "city_label": region_display_name(city),
        "total_in_city": int(len(subset)),
        "shown": len(points),
        "center": {"lat": center[0], "lng": center[1]},
        "bounds": {
            "south": min(lats),
            "north": max(lats),
            "west": min(lngs),
            "east": max(lngs),
        },
        "points": points,
    }


def load_model(model_path: Path | None = None) -> dict:
    model_path = model_path or MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Chưa tìm thấy mô hình tại {model_path}. "
            "Chạy `python src/train.py` để huấn luyện trước."
        )
    return joblib.load(model_path)


def format_vnd(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} tỷ"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f} triệu"
    return f"{value:,.0f} VND"


def format_vnd_full(value: float) -> str:
    return f"{value:,.0f} VND"
