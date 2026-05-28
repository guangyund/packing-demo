"""
软包材装箱引擎

公式三：
  pkg_L = stack_L + stack_H × SOFT_PKG_HEIGHT_FACTOR + SOFT_PKG_SEAL_BASE
  pkg_W = stack_W
  pkg_H = stack_H

货物按底面积从大到小垂直叠放，不使用 OR-Tools。
返回与 calculate_packing() 相同的数据结构（额外含 bin_data 字段供费率计算使用）。
"""
from config import SOFT_PKG_HEIGHT_FACTOR, SOFT_PKG_SEAL_BASE, SOFT_PKG_SIDE_MARGIN, SOFT_PKG_MAX_WEIGHT


def calc_soft_packing(items: list,
                       height_factor: float = None,
                       side_margin: float = None) -> dict | None:
    """
    计算软包材方案。超重时返回 None。

    Args:
        items:         货物列表，每项含 id/length/width/height/weight
        height_factor: 封口折叠比例，覆盖 config.SOFT_PKG_HEIGHT_FACTOR
        side_margin:   两侧余量（cm），覆盖 config.SOFT_PKG_SIDE_MARGIN
    Returns:
        包含 bin_data、packed_bins、summary 的 dict，或 None
    """
    height_factor = height_factor if height_factor is not None else SOFT_PKG_HEIGHT_FACTOR
    side_margin   = side_margin   if side_margin   is not None else SOFT_PKG_SIDE_MARGIN
    total_weight = sum(i.get("weight", 0) for i in items)
    if total_weight > SOFT_PKG_MAX_WEIGHT:
        return None

    def flat_dims(item):
        """平铺方向：三边降序，length=最长, width=中间, height=最小（厚度方向）"""
        d = sorted([item["length"], item["width"], item["height"]], reverse=True)
        return d[0], d[1], d[2]

    # ── 三种摆放策略，取体积最小者 ──────────────────────────────────────────────
    #
    # 策略1「竖叠」：所有货物沿 z（高度）方向依次叠放
    #   stack_h = sum(thickness)  → 高度最大，底面积最小
    #
    # 策略2「横排」：所有货物在 y（宽度）方向并排，单层摆平
    #   stack_h = max(thickness)  → 高度最小，宽度方向铺开
    #
    # 策略3「纵排」：所有货物在 x（长度）方向并排，单层摆平
    #   stack_h = max(thickness)  → 高度最小，长度方向铺开

    def _pkg_dims(sl, sw, sh):
        """根据堆叠包围盒计算软包材三边尺寸及体积"""
        pl = round(sl + sh * height_factor + SOFT_PKG_SEAL_BASE, 1)
        pw = round(sw + side_margin * 2, 1)
        ph = round(sh, 1)
        return pl, pw, ph, pl * pw * ph

    fl = [flat_dims(i) for i in items]
    max_l  = max(f[0] for f in fl)
    max_w  = max(f[1] for f in fl)
    max_h  = max(f[2] for f in fl)
    sum_l  = sum(f[0] for f in fl)
    sum_w  = sum(f[1] for f in fl)
    sum_h  = sum(f[2] for f in fl)

    strategies = [
        ("竖叠", max_l, max_w, sum_h),   # 高度=叠加厚度之和
        ("横排", max_l, sum_w, max_h),   # 沿宽度方向并排，高度=最大厚度
        ("纵排", sum_l, max_w, max_h),   # 沿长度方向并排，高度=最大厚度
    ]

    best_name, stack_l, stack_w, stack_h = min(
        strategies,
        key=lambda s: _pkg_dims(s[1], s[2], s[3])[3],   # 取包材体积最小的策略
    )

    pkg_l, pkg_w, pkg_h, pkg_vol = _pkg_dims(stack_l, stack_w, stack_h)

    # ── 货物摆放位置（用于 3D 展示） ─────────────────────────────────────────────
    sorted_items = sorted(
        items,
        key=lambda x: flat_dims(x)[0] * flat_dims(x)[1],
        reverse=True,
    )
    placed_items = []
    if best_name == "竖叠":
        # 沿 z 方向叠放
        z_off = 0.0
        for item in sorted_items:
            il, iw, ih = flat_dims(item)
            placed_items.append({
                "id":         item["id"],
                "position":   {"x": 0.0, "y": 0.0, "z": z_off},
                "dimensions": {"length": float(il), "width": float(iw), "height": float(ih)},
                "rotation_type": 0,
            })
            z_off += ih
    elif best_name == "横排":
        # 沿 y（宽度）方向并排，z=0 单层
        y_off = 0.0
        for item in sorted_items:
            il, iw, ih = flat_dims(item)
            placed_items.append({
                "id":         item["id"],
                "position":   {"x": 0.0, "y": y_off, "z": 0.0},
                "dimensions": {"length": float(il), "width": float(iw), "height": float(ih)},
                "rotation_type": 0,
            })
            y_off += iw
    else:  # 纵排
        # 沿 x（长度）方向并排，z=0 单层
        x_off = 0.0
        for item in sorted_items:
            il, iw, ih = flat_dims(item)
            placed_items.append({
                "id":         item["id"],
                "position":   {"x": x_off, "y": 0.0, "z": 0.0},
                "dimensions": {"length": float(il), "width": float(iw), "height": float(ih)},
                "rotation_type": 0,
            })
            x_off += il

    bbox_vol = stack_l * stack_w * stack_h
    util = round(bbox_vol / pkg_vol, 2) if pkg_vol > 0 else 0.0

    bin_data = {
        "type":       "软包材",
        "length":     pkg_l,
        "width":      pkg_w,
        "height":     pkg_h,
        "max_weight": float(SOFT_PKG_MAX_WEIGHT),
    }

    packed_bin = {
        "bin_type":    "软包材",
        "dimensions":  {"length": pkg_l, "width": pkg_w, "height": pkg_h},
        "utilization": util,
        "total_weight": round(total_weight, 3),
        "item_count":  len(items),
        "items":       placed_items,
    }

    return {
        "bin_data":       bin_data,
        "packed_bins":    [packed_bin],
        "unplaced_items": [],
        "summary": {
            "total_bins_used": 1,
            "avg_utilization": util,
            "all_placed":      True,
        },
    }
