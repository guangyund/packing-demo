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


def calc_soft_packing(items: list) -> dict | None:
    """
    计算软包材方案。超重时返回 None。

    Args:
        items: 货物列表，每项含 id/length/width/height/weight
    Returns:
        包含 bin_data、packed_bins、summary 的 dict，或 None
    """
    total_weight = sum(i.get("weight", 0) for i in items)
    if total_weight > SOFT_PKG_MAX_WEIGHT:
        return None

    def flat_dims(item):
        """平铺方向：三边降序，height 取最小（厚度方向）"""
        d = sorted([item["length"], item["width"], item["height"]], reverse=True)
        return d[0], d[1], d[2]

    sorted_items = sorted(
        items,
        key=lambda x: flat_dims(x)[0] * flat_dims(x)[1],
        reverse=True,
    )

    # 叠放包围盒
    stack_l = max(flat_dims(i)[0] for i in items)
    stack_w = max(flat_dims(i)[1] for i in items)
    stack_h = sum(flat_dims(i)[2] for i in items)

    # 软包材尺寸公式
    pkg_l = round(stack_l + stack_h * SOFT_PKG_HEIGHT_FACTOR + SOFT_PKG_SEAL_BASE, 1)
    pkg_w = round(stack_w + SOFT_PKG_SIDE_MARGIN * 2, 1)
    pkg_h = round(stack_h, 1)

    # 货物位置（垂直叠放，用于 3D 展示）
    placed_items = []
    z_offset = 0.0
    for item in sorted_items:
        il, iw, ih = flat_dims(item)
        placed_items.append({
            "id":         item["id"],
            "position":   {"x": 0.0, "y": 0.0, "z": z_offset},
            "dimensions": {"length": float(il), "width": float(iw), "height": float(ih)},
            "rotation_type": 0,
        })
        z_offset += ih

    pkg_vol  = pkg_l * pkg_w * pkg_h
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
