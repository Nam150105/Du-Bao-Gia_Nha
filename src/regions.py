"""34 đơn vị hành chính cấp tỉnh (Nghị quyết 202/2025/QH15, hiệu lực 12/6/2025)."""

from __future__ import annotations

import unicodedata

import pandas as pd

# 6 thành phố trực thuộc Trung ương
CENTRAL_CITIES: dict[str, str] = {
    "ha_noi": "Thành phố Hà Nội",
    "tp_ho_chi_minh": "Thành phố Hồ Chí Minh",
    "hai_phong": "Thành phố Hải Phòng",
    "hue": "Thành phố Huế",
    "da_nang": "Thành phố Đà Nẵng",
    "can_tho": "Thành phố Cần Thơ",
}

# 28 tỉnh
PROVINCES: dict[str, str] = {
    "an_giang": "An Giang",
    "bac_ninh": "Bắc Ninh",
    "ca_mau": "Cà Mau",
    "cao_bang": "Cao Bằng",
    "dak_lak": "Đắk Lắk",
    "dien_bien": "Điện Biên",
    "dong_nai": "Đồng Nai",
    "dong_thap": "Đồng Tháp",
    "gia_lai": "Gia Lai",
    "ha_tinh": "Hà Tĩnh",
    "hung_yen": "Hưng Yên",
    "khanh_hoa": "Khánh Hòa",
    "lai_chau": "Lai Châu",
    "lang_son": "Lạng Sơn",
    "lao_cai": "Lào Cai",
    "lam_dong": "Lâm Đồng",
    "nghe_an": "Nghệ An",
    "ninh_binh": "Ninh Bình",
    "phu_tho": "Phú Thọ",
    "quang_ngai": "Quảng Ngãi",
    "quang_ninh": "Quảng Ninh",
    "quang_tri": "Quảng Trị",
    "son_la": "Sơn La",
    "tay_ninh": "Tây Ninh",
    "thai_nguyen": "Thái Nguyên",
    "thanh_hoa": "Thanh Hóa",
    "tuyen_quang": "Tuyên Quang",
    "vinh_long": "Vĩnh Long",
}

REGION_OPTIONS: dict[str, str] = {**CENTRAL_CITIES, **PROVINCES}

# Một danh sách phẳng 34 tỉnh/thành (sắp A→Z theo tên hiển thị)
_REGION_LIST_SORTED = sorted(
    REGION_OPTIONS.items(),
    key=lambda item: item[1],
)
REGION_LIST: list[dict[str, str]] = [
    {"code": code, "name": name} for code, name in _REGION_LIST_SORTED
]

REGION_GROUPS: list[dict] = [
    {
        "id": "all",
        "label": "",
        "options": REGION_LIST,
    },
]

# Tên tỉnh/thành cũ (63 đơn vị) và biến thể trong CSV → mã mới
PROVINCE_ALIASES: dict[str, str] = {
    # Thành phố TWTU
    "Hà Nội": "ha_noi",
    "Thành phố Hà Nội": "ha_noi",
    "Hồ Chí Minh": "tp_ho_chi_minh",
    "Thành phố Hồ Chí Minh": "tp_ho_chi_minh",
    "TP. Hồ Chí Minh": "tp_ho_chi_minh",
    "TP HCM": "tp_ho_chi_minh",
    "Hải Phòng": "hai_phong",
    "Thành phố Hải Phòng": "hai_phong",
    "Huế": "hue",
    "Thành phố Huế": "hue",
    "Thừa Thiên Huế": "hue",
    "Thừa Thiên - Huế": "hue",
    "Đà Nẵng": "da_nang",
    "Thành phố Đà Nẵng": "da_nang",
    "Cần Thơ": "can_tho",
    "Thành phố Cần Thơ": "can_tho",
    # Tỉnh giữ nguyên / mới
    "An Giang": "an_giang",
    "Bắc Ninh": "bac_ninh",
    "Cà Mau": "ca_mau",
    "Cao Bằng": "cao_bang",
    "Đắk Lắk": "dak_lak",
    "Điện Biên": "dien_bien",
    "Đồng Nai": "dong_nai",
    "Đồng Tháp": "dong_thap",
    "Gia Lai": "gia_lai",
    "Hà Tĩnh": "ha_tinh",
    "Hưng Yên": "hung_yen",
    "Khánh Hòa": "khanh_hoa",
    "Lai Châu": "lai_chau",
    "Lạng Sơn": "lang_son",
    "Lào Cai": "lao_cai",
    "Lâm Đồng": "lam_dong",
    "Nghệ An": "nghe_an",
    "Ninh Bình": "ninh_binh",
    "Phú Thọ": "phu_tho",
    "Quảng Ngãi": "quang_ngai",
    "Quảng Ninh": "quang_ninh",
    "Quảng Trị": "quang_tri",
    "Sơn La": "son_la",
    "Tây Ninh": "tay_ninh",
    "Thái Nguyên": "thai_nguyen",
    "Thanh Hóa": "thanh_hoa",
    "Tuyên Quang": "tuyen_quang",
    "Vĩnh Long": "vinh_long",
    # Tỉnh cũ đã sáp nhập (2025)
    "Hà Giang": "tuyen_quang",
    "Yên Bái": "lao_cai",
    "Bắc Kạn": "thai_nguyen",
    "Vĩnh Phúc": "phu_tho",
    "Hòa Bình": "phu_tho",
    "Bắc Giang": "bac_ninh",
    "Hải Dương": "hai_phong",
    "Thái Bình": "hung_yen",
    "Hà Nam": "ninh_binh",
    "Nam Định": "ninh_binh",
    "Quảng Bình": "quang_tri",
    "Quảng Nam": "da_nang",
    "Kon Tum": "quang_ngai",
    "Bình Định": "gia_lai",
    "Phú Yên": "dak_lak",
    "Ninh Thuận": "khanh_hoa",
    "Đắk Nông": "lam_dong",
    "Bình Thuận": "lam_dong",
    "Bình Phước": "dong_nai",
    "Long An": "tay_ninh",
    "Bình Dương": "tp_ho_chi_minh",
    "Bà Rịa - Vũng Tàu": "tp_ho_chi_minh",
    "Bà Rịa-Vũng Tàu": "tp_ho_chi_minh",
    "Bà Rịa – Vũng Tàu": "tp_ho_chi_minh",
    "Tiền Giang": "dong_thap",
    "Kiên Giang": "an_giang",
    "Bến Tre": "vinh_long",
    "Trà Vinh": "vinh_long",
    "Sóc Trăng": "can_tho",
    "Hậu Giang": "can_tho",
    "Bạc Liêu": "ca_mau",
}

# Mã UI cũ (model đã train trước 2026) → mã mới
LEGACY_CITY_CODES: dict[str, str] = {
    "HCM": "tp_ho_chi_minh",
    "HN": "ha_noi",
    "DN": "da_nang",
    "BD": "tp_ho_chi_minh",
    "HY": "hung_yen",
    "HNA": "ninh_binh",
    "DNai": "dong_nai",
    "Khac": "khac",
}

# Tên hiển thị cũ trong bundle model
LEGACY_CITY_LABELS: dict[str, str] = {
    "HCM": "Hồ Chí Minh",
    "HN": "Hà Nội",
    "DN": "Đà Nẵng",
    "BD": "Bình Dương",
    "HY": "Hưng Yên",
    "HNA": "Hà Nam",
    "DNai": "Đồng Nai",
    "Khac": "Khác",
}

_NAME_TO_CODE: dict[str, str] = {}
for code, name in REGION_OPTIONS.items():
    _NAME_TO_CODE[name] = code
    _NAME_TO_CODE[name.replace("Thành phố ", "")] = code
for alias, code in PROVINCE_ALIASES.items():
    _NAME_TO_CODE[alias] = code

# Sắp theo độ dài tên giảm dần để khớp "Hà Nội" trước "Hà"
_SORTED_NAMES = sorted(_NAME_TO_CODE.keys(), key=len, reverse=True)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").lower()


def normalize_region_code(value: str) -> str | None:
    key = str(value or "").strip()
    if not key:
        return None
    if key in REGION_OPTIONS:
        return key
    if key in LEGACY_CITY_CODES:
        return LEGACY_CITY_CODES[key]
    if key in PROVINCE_ALIASES:
        return PROVINCE_ALIASES[key]
    if key in _NAME_TO_CODE:
        return _NAME_TO_CODE[key]
    return None


def resolve_city_for_model(city: str, model_city_options: dict) -> str:
    """Chọn mã city tương thích với mô hình đã lưu (hỗ trợ bundle cũ)."""
    code = normalize_region_code(city) or city
    if code in model_city_options:
        return code
    legacy = LEGACY_CITY_CODES.get(code) or LEGACY_CITY_CODES.get(city)
    if legacy and legacy in model_city_options:
        return legacy
    new_from_legacy = LEGACY_CITY_CODES.get(city)
    if new_from_legacy and new_from_legacy in model_city_options:
        return new_from_legacy
    for old_code, new_code in LEGACY_CITY_CODES.items():
        if new_code == code and old_code in model_city_options:
            return old_code
    raise ValueError(f"Khu vực không hợp lệ hoặc mô hình chưa hỗ trợ: {city}")


def extract_region_code(province: str | float, address: str | float = "") -> str:
    if pd.notna(province):
        key = str(province).strip()
        mapped = normalize_region_code(key)
        if mapped:
            return mapped
        if key in PROVINCE_ALIASES:
            return PROVINCE_ALIASES[key]
    text = str(address) if pd.notna(address) else ""
    if text:
        for name in _SORTED_NAMES:
            if name in text:
                return _NAME_TO_CODE[name]
    return "khac"


def region_display_name(code: str) -> str:
    if code in REGION_OPTIONS:
        return REGION_OPTIONS[code]
    if code in LEGACY_CITY_LABELS:
        return LEGACY_CITY_LABELS[code]
    if code == "khac":
        return "Khác / chưa xác định"
    return str(code)
