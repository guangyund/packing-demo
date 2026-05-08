"""
辅材推荐模块 - 根据产品品类和脆弱等级推荐包装辅材
"""

# 品类 → 脆弱等级（静态规则表，可扩展）
PRODUCT_FRAGILITY = {
    # 高脆弱度
    "手机": "high", "手机壳": "medium",
    "电脑": "high", "笔记本电脑": "high", "平板电脑": "high",
    "陶瓷": "high", "瓷器": "high", "陶器": "high",
    "玻璃": "high", "玻璃杯": "high", "玻璃瓶": "high",
    "消费电子": "high", "电子产品": "high",
    "屏幕": "high", "显示器": "high", "电视": "high",
    "相机": "high", "镜头": "high", "摄像头": "high",
    "音箱": "high", "耳机": "high",
    "钟表": "high", "手表": "high",
    "灯具": "high", "灯泡": "high",
    # 中等脆弱度
    "食品": "medium", "零食": "medium", "饮料": "medium",
    "玩具": "medium", "模型": "medium",
    "家具": "medium", "木质家具": "medium",
    "工具": "medium", "五金": "medium",
    "厨具": "medium", "锅具": "medium",
    "化妆品": "medium", "护肤品": "medium",
    "鞋子": "medium", "运动鞋": "medium",
    "包包": "medium", "箱包": "medium",
    "数码配件": "medium", "充电器": "medium",
    # 低脆弱度
    "服装": "low", "衣服": "low", "裤子": "low", "上衣": "low",
    "书籍": "low", "图书": "low", "杂志": "low",
    "文具": "low", "笔": "low", "本子": "low",
    "布料": "low", "纺织品": "low",
    "塑料制品": "low",
    "纸制品": "low",
}

# 脆弱等级 → 辅材组合（厚度单位 mm，价格单位 元/件或元/套）
MATERIAL_RULES = {
    "high": [
        {
            "type": "珍珠棉内衬",
            "thickness_mm": 20,
            "sides": "all",
            "unit_price": 3.0,
            "description": "四面+上下各20mm珍珠棉，提供全方位缓冲保护",
        },
        {
            "type": "气泡膜缠绕",
            "wrap_layers": 2,
            "add_per_side_cm": 0.5,
            "unit_price": 2.0,
            "description": "双层气泡膜缠绕，防震防撞",
        },
        {
            "type": "纸护角",
            "quantity": 8,
            "unit_price": 0.8,
            "description": "8个纸护角保护产品四角，防止碰角损坏",
        },
    ],
    "medium": [
        {
            "type": "珍珠棉内衬",
            "thickness_mm": 10,
            "sides": "all",
            "unit_price": 1.5,
            "description": "四面+上下各10mm珍珠棉，基础缓冲保护",
        },
        {
            "type": "气柱袋",
            "add_per_side_cm": 2.0,
            "unit_price": 3.0,
            "description": "气柱袋充气防震，适合中等易碎产品",
        },
    ],
    "low": [
        {
            "type": "纸屑填充",
            "add_per_side_cm": 1.0,
            "unit_price": 0.5,
            "description": "纸屑填充空隙，防止产品在箱内移动",
        },
    ],
}


def get_fragility(category: str, ai_fragility: str) -> str:
    """
    根据品类查询脆弱等级，查不到则用 AI 返回的值兜底。

    Args:
        category: 产品品类名称
        ai_fragility: AI 识别返回的脆弱等级（high/medium/low）

    Returns:
        脆弱等级字符串 ("high" / "medium" / "low")
    """
    # 精确匹配
    if category in PRODUCT_FRAGILITY:
        return PRODUCT_FRAGILITY[category]

    # 模糊匹配（包含关系）
    for key, level in PRODUCT_FRAGILITY.items():
        if key in category or category in key:
            return level

    # 兜底：使用 AI 返回值，若无效则默认 medium
    if ai_fragility in ("high", "medium", "low"):
        return ai_fragility
    return "medium"


def recommend_materials(fragility: str) -> list:
    """
    根据脆弱等级返回推荐辅材列表。

    Args:
        fragility: 脆弱等级 ("high" / "medium" / "low")

    Returns:
        辅材列表，每项为字典
    """
    rules = MATERIAL_RULES.get(fragility, MATERIAL_RULES["medium"])
    return [dict(m) for m in rules]  # 返回副本，避免外部修改影响规则表


def calc_packed_dimensions(dims: dict, materials: list) -> dict:
    """
    根据产品净尺寸和辅材，计算加辅材后的实际装箱尺寸。

    对于有 thickness_mm 的辅材（珍珠棉等），每面各加一次厚度（即总共加 thickness_mm×2）。
    对于有 add_per_side_cm 的辅材（气泡膜、气柱袋等），每面各加一次（总共加 add_per_side_cm×2）。

    Args:
        dims: {"length": float, "width": float, "height": float}（cm）
        materials: recommend_materials 的返回值

    Returns:
        {"length": float, "width": float, "height": float}（cm，已加辅材）
    """
    extra_cm = 0.0  # 每个方向的单侧增量（cm）

    for mat in materials:
        if "thickness_mm" in mat:
            extra_cm += mat["thickness_mm"] / 10.0  # mm → cm
        if "add_per_side_cm" in mat:
            extra_cm += mat["add_per_side_cm"]

    # 每面两侧都要加，故每个维度增加 extra_cm * 2
    total_extra = extra_cm * 2
    return {
        "length": round(dims["length"] + total_extra, 2),
        "width":  round(dims["width"]  + total_extra, 2),
        "height": round(dims["height"] + total_extra, 2),
    }


def calc_material_cost(materials: list) -> float:
    """
    计算辅材总成本（元）。

    Args:
        materials: recommend_materials 的返回值

    Returns:
        辅材总成本（元），保留2位小数
    """
    total = sum(m.get("unit_price", 0.0) for m in materials)
    return round(total, 2)
