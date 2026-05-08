"""
运费档级计算模块

档级判断规则（尺寸单位 cm，重量单位 g；L≥W≥H 为排序后三边）：
  小号标准件   L≤36, W≤28, H≤1.6, 实际重≤454g
  大号标准件   L≤43, W≤34, H≤19, 实际重≤9080g
  小号大件     L≤92, W≤70, H≤49, L+围长≤330, 实际重≤22680g
  大号大件     L≤147, W≤81, H≤81, L+围长≤330, 实际重≤22680g, 计费重≤22675g
  超大件       L>147(≤241) 或 W/H>81(≤241) 或 L+围长>330 或 计费重>22675g
  特大件       任一边>241cm

围长 = 2×(次长边+最短边)
体积重(g) = L(cm)×W(cm)×H(cm) / 5  ← 等价于用户公式 (长×宽×高)/5000 kg ×1000
计费重(g) = max(实际重量, 体积重)

产品类别：常规类产品 / 服装产品 / 危险品
售价区间：≤$10 / $10-$50 / >$50

费率行结构：(重量上限g, 重量依据, [base≤10, base10-50, base>50], 超出阈值g, 每步g, 每步费)
  - 重量依据："actual"=实际重，"billable"=计费重
  - 超出阈值=0 表示纯固定费率，无阶梯部分
"""

import math

DIM_FACTOR = 5.0  # cm³→g 换算系数（等价于体积重kg = L×W×H/5000）

# ── 档级名称 ───────────────────────────────────────────────────────────────────
TIER_SMALL_STANDARD    = "小号标准件"
TIER_LARGE_STANDARD    = "大号标准件"
TIER_SMALL_OVERSIZED   = "小号大件"
TIER_LARGE_OVERSIZED   = "大号大件"
TIER_SPECIAL_OVERSIZED = "超大件"
TIER_EXTRA_LARGE       = "特大件"

TIER_ORDER = [
    TIER_SMALL_STANDARD,
    TIER_LARGE_STANDARD,
    TIER_SMALL_OVERSIZED,
    TIER_LARGE_OVERSIZED,
    TIER_SPECIAL_OVERSIZED,
    TIER_EXTRA_LARGE,
]

# ── 产品类别 ───────────────────────────────────────────────────────────────────
CATEGORY_STANDARD = "常规类产品"
CATEGORY_APPAREL  = "服装产品"
CATEGORY_HAZMAT   = "危险品"
PRODUCT_CATEGORIES = [CATEGORY_STANDARD, CATEGORY_APPAREL, CATEGORY_HAZMAT]

# ── 售价区间（USD）──────────────────────────────────────────────────────────────
PRICE_BRACKETS = [
    (0,   10,           "≤$10"),
    (10,  50,           "$10-$50"),
    (50,  float("inf"), ">$50"),
]

def _price_idx(sale_price_usd: float) -> int:
    for i, (lo, hi, _) in enumerate(PRICE_BRACKETS):
        if lo <= sale_price_usd < hi:
            return i
    return len(PRICE_BRACKETS) - 1


# ── 运费表 ─────────────────────────────────────────────────────────────────────
# 每条费率行：(重量上限g, 重量依据, [base≤10,base10-50,base>50], 超出阈值g, 每步g, 每步费)
# 超出阈值=0 → 纯固定费；否则：费用 = base + ceil((weight-阈值)/每步) × 每步费
# inf = 不限重量

_INF = float("inf")
_B   = "billable"  # 计费重量（max 实际重, 体积重）
_A   = "actual"    # 实际重量

SHIPPING_FEE_TABLE: dict[str, dict[str, list[tuple]]] = {

    # ── 小号标准件（按实际重量查档，纯固定费）────────────────────────────────────
    TIER_SMALL_STANDARD: {
        CATEGORY_STANDARD: [
            ( 56.75, _A, [2.43, 3.32, 3.58], 0, 0, 0),
            (113.5,  _A, [2.49, 3.42, 3.68], 0, 0, 0),
            (170.25, _A, [2.56, 3.45, 3.71], 0, 0, 0),
            (227.0,  _A, [2.66, 3.54, 3.80], 0, 0, 0),
            (283.75, _A, [2.77, 3.68, 3.94], 0, 0, 0),
            (340.5,  _A, [2.82, 3.78, 4.04], 0, 0, 0),
            (397.25, _A, [2.92, 3.91, 4.17], 0, 0, 0),
            (454.0,  _A, [2.95, 3.96, 4.22], 0, 0, 0),
        ],
        CATEGORY_APPAREL: [
            ( 56.75, _A, [2.62, 3.51, 3.77], 0, 0, 0),
            (113.5,  _A, [2.64, 3.54, 3.80], 0, 0, 0),
            (170.25, _A, [2.68, 3.59, 3.85], 0, 0, 0),
            (227.0,  _A, [2.81, 3.69, 3.95], 0, 0, 0),
            (283.75, _A, [3.00, 3.91, 4.17], 0, 0, 0),
            (340.5,  _A, [3.10, 4.09, 4.35], 0, 0, 0),
            (397.25, _A, [3.20, 4.20, 4.46], 0, 0, 0),
            (454.0,  _A, [3.30, 4.25, 4.51], 0, 0, 0),
        ],
        CATEGORY_HAZMAT: [
            ( 56.75, _A, [3.40, 4.29, 4.55], 0, 0, 0),
            (113.5,  _A, [3.43, 4.36, 4.62], 0, 0, 0),
            (170.25, _A, [3.48, 4.37, 4.63], 0, 0, 0),
            (227.0,  _A, [3.55, 4.43, 4.69], 0, 0, 0),
            (283.75, _A, [3.64, 4.55, 4.81], 0, 0, 0),
            (340.5,  _A, [3.65, 4.61, 4.87], 0, 0, 0),
            (397.25, _A, [3.73, 4.72, 4.98], 0, 0, 0),
            (454.0,  _A, [3.77, 4.78, 5.04], 0, 0, 0),
        ],
    },

    # ── 大号标准件（按计费重查档）────────────────────────────────────────────────
    # 前12档固定，最后1档阶梯（标准/危险品每113.5g+0.08，服装每227g+0.16）
    TIER_LARGE_STANDARD: {
        CATEGORY_STANDARD: [
            ( 113.5,  _B, [2.91, 3.73, 3.99], 0,    0,     0   ),
            ( 227.0,  _B, [3.13, 3.95, 4.21], 0,    0,     0   ),
            ( 340.5,  _B, [3.38, 4.20, 4.46], 0,    0,     0   ),
            ( 454.0,  _B, [3.78, 4.60, 4.86], 0,    0,     0   ),
            ( 567.5,  _B, [4.22, 5.04, 5.30], 0,    0,     0   ),
            ( 681.0,  _B, [4.60, 5.42, 5.68], 0,    0,     0   ),
            ( 794.5,  _B, [4.75, 5.57, 5.83], 0,    0,     0   ),
            ( 908.0,  _B, [5.00, 5.82, 6.08], 0,    0,     0   ),
            (1021.5,  _B, [5.10, 5.92, 6.18], 0,    0,     0   ),
            (1135.0,  _B, [5.28, 6.10, 6.36], 0,    0,     0   ),
            (1248.5,  _B, [5.44, 6.26, 6.52], 0,    0,     0   ),
            (1362.0,  _B, [5.85, 6.67, 6.93], 0,    0,     0   ),
            (9080.0,  _B, [6.15, 6.97, 6.82], 1362, 113.5, 0.08),
        ],
        CATEGORY_APPAREL: [
            ( 113.5,  _B, [3.48, 4.30, 4.56], 0,    0,    0   ),
            ( 227.0,  _B, [3.68, 4.50, 4.76], 0,    0,    0   ),
            ( 340.5,  _B, [3.90, 4.72, 4.98], 0,    0,    0   ),
            ( 454.0,  _B, [4.35, 5.17, 5.43], 0,    0,    0   ),
            ( 567.5,  _B, [5.05, 5.87, 6.13], 0,    0,    0   ),
            ( 681.0,  _B, [5.22, 6.04, 6.30], 0,    0,    0   ),
            ( 794.5,  _B, [5.32, 6.14, 6.40], 0,    0,    0   ),
            ( 908.0,  _B, [5.43, 6.25, 6.51], 0,    0,    0   ),
            (1021.5,  _B, [5.78, 6.60, 6.86], 0,    0,    0   ),
            (1135.0,  _B, [5.90, 6.72, 6.98], 0,    0,    0   ),
            (1248.5,  _B, [5.95, 6.77, 7.03], 0,    0,    0   ),
            (1362.0,  _B, [6.08, 6.90, 7.16], 0,    0,    0   ),
            (9080.0,  _B, [6.82, 6.97, 7.63], 1362, 227,  0.16),
        ],
        CATEGORY_HAZMAT: [
            ( 113.5,  _B, [3.73, 4.55, 4.81], 0,    0,     0   ),
            ( 227.0,  _B, [3.94, 4.76, 5.02], 0,    0,     0   ),
            ( 340.5,  _B, [4.17, 4.99, 5.25], 0,    0,     0   ),
            ( 454.0,  _B, [4.37, 5.19, 5.45], 0,    0,     0   ),
            ( 567.5,  _B, [4.82, 5.64, 5.90], 0,    0,     0   ),
            ( 681.0,  _B, [5.20, 6.02, 6.28], 0,    0,     0   ),
            ( 794.5,  _B, [5.35, 6.17, 6.43], 0,    0,     0   ),
            ( 908.0,  _B, [5.49, 6.31, 6.57], 0,    0,     0   ),
            (1021.5,  _B, [5.56, 6.38, 6.64], 0,    0,     0   ),
            (1135.0,  _B, [5.74, 6.56, 6.82], 0,    0,     0   ),
            (1248.5,  _B, [5.90, 6.72, 6.98], 0,    0,     0   ),
            (1362.0,  _B, [6.31, 7.13, 7.39], 0,    0,     0   ),
            (9080.0,  _B, [6.61, 7.43, 7.69], 1362, 113.5, 0.08),
        ],
    },

    # ── 小号大件（计费重，阶梯：超过454g每454g多收0.38）────────────────────────────
    TIER_SMALL_OVERSIZED: {
        CATEGORY_STANDARD: [
            (22675, _B, [6.78, 7.55, 7.55], 454, 454, 0.38),
        ],
        CATEGORY_APPAREL: [
            (22675, _B, [6.78, 7.55, 7.55], 454, 454, 0.38),
        ],
        CATEGORY_HAZMAT: [
            (22675, _B, [7.50, 8.27, 8.27], 454, 454, 0.38),
        ],
    },

    # ── 大号大件（计费重，阶梯：超过454g每454g多收0.38）────────────────────────────
    TIER_LARGE_OVERSIZED: {
        CATEGORY_STANDARD: [
            (22675, _B, [8.58, 9.35,  9.35 ], 454, 454, 0.38),
        ],
        CATEGORY_APPAREL: [
            (22675, _B, [8.58, 9.35,  9.35 ], 454, 454, 0.38),
        ],
        CATEGORY_HAZMAT: [
            (22675, _B, [9.30, 10.07, 10.07], 454, 454, 0.38),
        ],
    },

    # ── 超大件（三段计费重阶梯 + 一段实际重不限）─────────────────────────────────
    TIER_SPECIAL_OVERSIZED: {
        CATEGORY_STANDARD: [
            (22675, _B, [25.56,  26.33,  26.33 ], 454,   454, 0.38),
            (31745, _B, [36.55,  37.32,  37.32 ], 23130, 454, 0.75),
            (68038, _B, [50.55,  51.32,  51.32 ], 32204, 454, 0.75),
            (_INF,  _A, [194.18, 194.95, 194.95], 68487, 454, 0.19),
        ],
        CATEGORY_APPAREL: [
            (22675, _B, [25.56,  26.33,  26.33 ], 454,   454, 0.38),
            (31745, _B, [36.55,  37.32,  37.32 ], 23130, 454, 0.75),
            (68038, _B, [50.55,  51.32,  51.32 ], 32204, 454, 0.75),
            (_INF,  _A, [194.18, 194.95, 194.95], 68487, 454, 0.19),
        ],
        CATEGORY_HAZMAT: [
            (22675, _B, [27.67,  28.44,  28.44 ], 454,   454, 0.38),
            (31745, _B, [39.76,  40.53,  40.53 ], 23130, 454, 0.75),
            (68038, _B, [57.68,  58.45,  58.45 ], 32204, 454, 0.75),
            (_INF,  _A, [218.76, 219.53, 219.53], 68487, 454, 0.19),
        ],
    },

    # ── 特大件（与超大件费率相同，档级由尺寸决定：任一边>241cm）──────────────────
    TIER_EXTRA_LARGE: {
        CATEGORY_STANDARD: [
            (22675, _B, [25.56,  26.33,  26.33 ], 454,   454, 0.38),
            (31745, _B, [36.55,  37.32,  37.32 ], 23130, 454, 0.75),
            (68038, _B, [50.55,  51.32,  51.32 ], 32204, 454, 0.75),
            (_INF,  _A, [194.18, 194.95, 194.95], 68487, 454, 0.19),
        ],
        CATEGORY_APPAREL: [
            (22675, _B, [25.56,  26.33,  26.33 ], 454,   454, 0.38),
            (31745, _B, [36.55,  37.32,  37.32 ], 23130, 454, 0.75),
            (68038, _B, [50.55,  51.32,  51.32 ], 32204, 454, 0.75),
            (_INF,  _A, [194.18, 194.95, 194.95], 68487, 454, 0.19),
        ],
        CATEGORY_HAZMAT: [
            (22675, _B, [27.67,  28.44,  28.44 ], 454,   454, 0.38),
            (31745, _B, [39.76,  40.53,  40.53 ], 23130, 454, 0.75),
            (68038, _B, [57.68,  58.45,  58.45 ], 32204, 454, 0.75),
            (_INF,  _A, [218.76, 219.53, 219.53], 68487, 454, 0.19),
        ],
    },
}


# ── 包装费表 ───────────────────────────────────────────────────────────────────
# 小号大件 / 大号大件：共用，按「体积重量」查询
_OVERSIZED_PKG = [
    (2267,  1.51), (4535,  1.68), (6803,  1.97), (9070,  2.60),
    (11339, 2.92), (13607, 3.47), (15875, 3.60), (18143, 3.78),
    (20411, 3.80), (_INF,  4.04),
]

PACKAGING_FEE_TABLE: dict[str, list[tuple]] = {
    TIER_SMALL_OVERSIZED: _OVERSIZED_PKG,
    TIER_LARGE_OVERSIZED: _OVERSIZED_PKG,
}

# 超大件：按「实际重量」查询
SPECIAL_OVERSIZED_PKG_TABLE = [
    (22675, 17.0),
    (31745, 21.0),
    (68038, 25.0),
]


# ── 查表工具函数 ───────────────────────────────────────────────────────────────

def _lookup_flat(table: list[tuple], key_g: float) -> float:
    for max_g, val in table:
        if key_g <= max_g:
            return val
    return table[-1][1]


def _calc_row_fee(row: tuple, actual_g: float, billable_g: float, pidx: int) -> float:
    """根据费率行计算运费（固定 or 固定+阶梯）"""
    _max_wt, wt_basis, bases, threshold, step, step_fee = row
    base = bases[pidx]
    if threshold == 0:
        return base
    lookup_w = actual_g if wt_basis == _A else billable_g
    if lookup_w <= threshold:
        return base
    extra_units = math.ceil((lookup_w - threshold) / step)
    return base + extra_units * step_fee


def _find_fee_row(rows: list[tuple], actual_g: float, billable_g: float) -> tuple | None:
    """选出匹配的费率行：先按计费重匹配有限段，再按实际重匹配不限段"""
    for row in rows:
        max_wt, wt_basis, *_ = row
        if wt_basis == _B and billable_g <= max_wt:
            return row
    for row in rows:
        max_wt, wt_basis, *_ = row
        if wt_basis == _A:
            return row
    return None


# ── 档级分类 ───────────────────────────────────────────────────────────────────

def classify_tier(length_cm: float, width_cm: float, height_cm: float,
                  actual_weight_g: float) -> tuple[str, float]:
    L, W, H  = sorted([length_cm, width_cm, height_cm], reverse=True)
    girth    = 2 * (W + H)
    lwg      = L + girth
    dim_wt_g = L * W * H / DIM_FACTOR
    billable = max(actual_weight_g, dim_wt_g)

    # 特大件：任一边超过 241cm
    if L > 241:
        return TIER_EXTRA_LARGE, billable

    # 小号标准件
    if L <= 36 and W <= 28 and H <= 1.6 and actual_weight_g <= 454:
        return TIER_SMALL_STANDARD, billable

    # 大号标准件
    if L <= 43 and W <= 34 and H <= 19 and actual_weight_g <= 9080:
        return TIER_LARGE_STANDARD, billable

    # 小号大件
    if L <= 92 and W <= 70 and H <= 49 and lwg <= 330 and actual_weight_g <= 22680:
        return TIER_SMALL_OVERSIZED, billable

    # 大号大件（计费重须 ≤ 22675g）
    if (L <= 147 and W <= 81 and H <= 81 and lwg <= 330
            and actual_weight_g <= 22680 and billable <= 22675):
        return TIER_LARGE_OVERSIZED, billable

    # 超大件（兜底）
    return TIER_SPECIAL_OVERSIZED, billable


def tier_rank(tier_name: str) -> int:
    try:
        return TIER_ORDER.index(tier_name)
    except ValueError:
        return len(TIER_ORDER)


# ── 费用计算 ───────────────────────────────────────────────────────────────────

def calc_total_fee(tier: str, billable_weight_g: float,
                   actual_weight_g: float, dim_weight_g: float,
                   sale_price_usd: float,
                   product_category: str = CATEGORY_STANDARD) -> dict:
    """
    Args:
        billable_weight_g: max(actual, dim)
        actual_weight_g:   实际重量
        dim_weight_g:      体积重量（L×W×H/5）
    """
    pidx = _price_idx(sale_price_usd)

    tier_table = SHIPPING_FEE_TABLE.get(tier, {})
    rows = tier_table.get(product_category) or tier_table.get(CATEGORY_STANDARD, [])
    row  = _find_fee_row(rows, actual_weight_g, billable_weight_g)
    shipping_fee = _calc_row_fee(row, actual_weight_g, billable_weight_g, pidx) if row else 0.0

    # 小号/大号大件包装费：按体积重查
    packaging_fee = 0.0
    if tier in PACKAGING_FEE_TABLE:
        packaging_fee = _lookup_flat(PACKAGING_FEE_TABLE[tier], dim_weight_g)

    # 超大件包装费：按实际重查
    if tier == TIER_SPECIAL_OVERSIZED:
        packaging_fee = _lookup_flat(SPECIAL_OVERSIZED_PKG_TABLE, actual_weight_g)

    return {
        "tier":             tier,
        "product_category": product_category,
        "shipping_fee":     round(shipping_fee, 2),
        "packaging_fee":    round(packaging_fee, 2),
        "total_fee":        round(shipping_fee + packaging_fee, 2),
    }


def calc_bin_fee(bin_dims: dict, total_weight_kg: float,
                 sale_price_usd: float,
                 product_category: str = CATEGORY_STANDARD) -> dict:
    actual_g  = total_weight_kg * 1000
    L, W, H   = bin_dims["length"], bin_dims["width"], bin_dims["height"]
    dim_wt_g  = L * W * H / DIM_FACTOR
    tier, billable_g = classify_tier(L, W, H, actual_g)
    fee_info = calc_total_fee(tier, billable_g, actual_g, dim_wt_g, sale_price_usd, product_category)
    fee_info["billable_weight_g"] = round(billable_g, 1)
    fee_info["actual_weight_g"]   = round(actual_g, 1)
    fee_info["dim_weight_g"]      = round(dim_wt_g, 1)
    return fee_info
