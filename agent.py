"""
三维装箱 AI Agent
使用 Claude 作为决策大脑，自动选择箱型、调用算法、验证方案。
当箱型库没有合适箱型时，自动推荐新包材并与最优现有箱型对比。
"""
import json
import math
import time
import logging
import httpx
import anthropic

logger = logging.getLogger(__name__)
from packing_engine import calculate_packing, AVAILABLE_BINS, MAX_BIN_WEIGHT
from config import (MAX_BIN_LONG_SIDE, MAX_BIN_SHORT_SIDE, BIN_SIZE_BUFFER, BIN_SIZE_ROUND_TO,
                    MIN_UTILIZATION_THRESHOLD)
from shipping_tiers import calc_bin_fee, tier_rank, TIER_ORDER
from soft_packing_engine import calc_soft_packing

client = anthropic.Anthropic()

# ── 包材目录（自动从 bins_catalog.json 加载）──────────────────────────────────
import os as _os
_CATALOG_PATH = _os.path.join(_os.path.dirname(__file__), "bins_catalog.json")
try:
    with open(_CATALOG_PATH, encoding="utf-8") as _f:
        _RAW_CATALOG: list = json.load(_f)
except (FileNotFoundError, Exception):
    _RAW_CATALOG = []

print(f"[agent] 包材目录已加载：共 {len(_RAW_CATALOG)} 条 | TOOLS数量: 稍后显示", flush=True)


def _catalog_to_bin(c: dict) -> dict:
    name = (c.get("mat_name") or "").strip() or c["name"][:15]
    return {
        "type":       f"{name} ({c['sku']})" if c.get("sku") else name,
        "length":     float(c["length"]),
        "width":      float(c["width"]),
        "height":     float(c["height"]),
        "max_weight": float(c.get("max_weight", 22)),
        "cost_price": float(c.get("price", 0)),
        "sku":        c.get("sku", ""),
    }


def _prefilter_catalog_bins(items: list, max_results: int = 150) -> list:
    """
    从目录中筛选适合当前货物的硬包材候选箱型，按体积升序排列。
    筛选条件：
      1. 箱型承重 >= 货物总重
      2. 箱型体积 >= 货物总体积
      3. 每件货物（旋转后）均能放入箱中（三边升序逐一对比）
    """
    if not _RAW_CATALOG:
        return AVAILABLE_BINS
    total_weight = sum(i.get("weight", 0) for i in items)
    total_vol    = sum(i["length"] * i["width"] * i["height"] for i in items)
    item_sorted_dims = [sorted([i["length"], i["width"], i["height"]]) for i in items]

    candidates = []
    for c in _RAW_CATALOG:
        if c.get("type") != "硬包材":
            continue
        if c.get("max_weight", 22) < total_weight:
            continue
        bin_vol = c["length"] * c["width"] * c["height"]
        if bin_vol < total_vol:
            continue
        b_s = sorted([c["length"], c["width"], c["height"]])
        if all(ids[0] <= b_s[0] and ids[1] <= b_s[1] and ids[2] <= b_s[2]
               for ids in item_sorted_dims):
            candidates.append(c)

    candidates.sort(key=lambda c: c["length"] * c["width"] * c["height"])
    if not candidates:
        logger.warning("目录预筛选结果为空，退回 AVAILABLE_BINS（货物总重%.2fkg，总体积%.0fcm³）",
                       total_weight, total_vol)
        return AVAILABLE_BINS
    logger.info("目录预筛选：%d → %d 个候选硬包材", len(_RAW_CATALOG), min(len(candidates), max_results))
    return [_catalog_to_bin(c) for c in candidates[:max_results]]


# 各档位边界包材（尺寸均满足该档位所有约束），用于降档可行性验证
# max_weight 单位 kg（与装箱引擎一致）
# 注：大号大件因需同时满足 lwg≤330 且体积重≤22675g，约束较复杂，暂不列入
_TIER_BOUNDARY_BOXES = [
    # (tier_name, bin_dict) 从低档到高档排列
    ("大号标准件", {"type": "推荐新包材", "length": 43.0, "width": 34.0, "height": 19.0, "max_weight": 9.08}),
    # 小号大件：lwg = 92+2*(70+49)=330 恰好满足 ≤330；体积重=92*70*49/5=63,112g 超 22675g，但实际
    # 装箱时重量通常远低于此，classify_tier 基于 actual_weight 约束（≤22680g）判定，可正常降档
    ("小号大件",   {"type": "推荐新包材", "length": 92.0, "width": 70.0, "height": 49.0, "max_weight": 22.68}),
]

# ── 工具定义 ──────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "recommend_and_compare",
        "description": (
            "三维装箱主工具。自动从包材目录（4000+种）中扫描最优硬包材，"
            "同时评估软包材方案，输出推荐包材、利用率、FBA费档级对比结果。"
            "直接传入货物列表即可，无需提前查询箱型库。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "完整货物列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id":                {"type": "string"},
                            "length":            {"type": "number"},
                            "width":             {"type": "number"},
                            "height":            {"type": "number"},
                            "weight":            {"type": "number"},
                            "soft_packaging_ok": {"type": "boolean"},
                        },
                        "required": ["id", "length", "width", "height", "weight"],
                    },
                },
            },
            "required": ["items"],
        },
    },
]

print(f"[agent] TOOLS 已注册：{[t['name'] for t in TOOLS]}", flush=True)

# ── 推荐新包材算法 ────────────────────────────────────────────────────────────

def _calc_recommended_bin(items: list) -> dict:
    """
    通过垂直叠放估算货物最小占用空间，得出推荐包材尺寸：

    1. 按系统重量上限将货物分批
    2. 每批按底面积从大到小垂直叠放：
       - L / W 取各件平铺后的最大值（不横向累加）
       - H 逐件累加（高度方向叠放）
       这样保证货物紧密接邻、不横向散铺，推荐出更合身的包材
    3. 取各批最大包围盒，加缓冲余量后向上取整
    """
    def _flat_dims(item):
        """平铺时的 (L, W, H)：三边降序，H 最小"""
        d = sorted([item["length"], item["width"], item["height"]], reverse=True)
        return d[0], d[1], d[2]

    # 按体积从大到小排序，大件优先
    sorted_items = sorted(
        items, key=lambda x: x["length"] * x["width"] * x["height"], reverse=True
    )

    # 按重量上限分批
    batches: list = []
    current: list = []
    cur_w = 0.0
    for item in sorted_items:
        if current and cur_w + item["weight"] > MAX_BIN_WEIGHT:
            batches.append(current)
            current = [item]
            cur_w   = item["weight"]
        else:
            current.append(item)
            cur_w += item["weight"]
    if current:
        batches.append(current)

    # 对每批垂直叠放，量出包围盒
    max_l = max_w = max_h = 0.0
    for batch in batches:
        # 按底面积从大到小：大件在底，小件叠上
        batch_sorted = sorted(
            batch, key=lambda i: _flat_dims(i)[0] * _flat_dims(i)[1], reverse=True
        )
        bl = bw = bh = 0.0
        for item in batch_sorted:
            l, w, h = _flat_dims(item)
            bl  = max(bl, l)   # 底面长取最大值
            bw  = max(bw, w)   # 底面宽取最大值
            bh += h            # 高度方向累加叠放
        max_l = max(max_l, bl)
        max_w = max(max_w, bw)
        max_h = max(max_h, bh)

    # 兜底
    if max_l == 0:
        max_l = max(i["length"] for i in items)
        max_w = max(i["width"]  for i in items)
        max_h = max(i["height"] for i in items)

    # 加缓冲余量，向上取整到指定精度
    rec_l = math.ceil((max_l + BIN_SIZE_BUFFER) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO
    rec_w = math.ceil((max_w + BIN_SIZE_BUFFER) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO
    rec_h = math.ceil((max_h + BIN_SIZE_BUFFER) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO

    # 应用包材尺寸上限：最长边 ≤ MAX_BIN_LONG_SIDE，另外两边 ≤ MAX_BIN_SHORT_SIDE
    dims = sorted([rec_l, rec_w, rec_h], reverse=True)
    dims[0] = min(dims[0], MAX_BIN_LONG_SIDE)
    dims[1] = min(dims[1], MAX_BIN_SHORT_SIDE)
    dims[2] = min(dims[2], MAX_BIN_SHORT_SIDE)
    rec_l, rec_w, rec_h = dims

    return {
        "type":       "推荐新包材",
        "length":     float(rec_l),
        "width":      float(rec_w),
        "height":     float(rec_h),
        "max_weight": float(MAX_BIN_WEIGHT),
    }


def _do_recommend_and_compare(items: list, bins: list = None) -> tuple:
    """
    推荐新包材并与箱型库最优结果对比。
    两种方案均使用「多箱」策略，直到所有货物全部装完。

    新增运费档级比较：
      - 若推荐包材能达到更低的运费档级，不论利用率如何都强制推荐
      - 现有包材已单箱装完且利用率≥80% 的早返回门槛，也会被更低档级覆盖

    Returns:
        (agent_summary_str, full_compare_result, primary_result)
        primary_result 始终是 winner 对应的完整装箱数据
    """
    available = bins if bins else (_prefilter_catalog_bins(items) if _RAW_CATALOG else AVAILABLE_BINS)
    max_copies = len(items)
    print(f"[recommend] 扫描包材数: {len(available)}, 前3: {[b['type'] for b in available[:3]]}", flush=True)

    # 提取货物信息：总重(kg)、售价(USD)、产品类别
    total_weight_kg  = sum(i.get("weight", 0) for i in items)
    sale_price_usd   = max((i.get("sale_price", 0) for i in items), default=0)
    product_category = next((i.get("product_category", "常规类产品") for i in items
                             if i.get("product_category")), "常规类产品")

    # ── 第一步：扫描现有箱型，找出最优（贪心快速评分，不运行 OR-Tools）─────────────
    # scan_mode=True：跳过 OR-Tools，仅用贪心算法给每个箱型打分（< 100ms/箱型）。
    # 确定最优箱型后，再单独跑一次完整 OR-Tools 取得精确坐标用于 3D 展示。
    best_bin_result = None
    best_full       = None
    best_score      = float("-inf")
    best_bin_data   = None

    for bin_data in available:
        result     = calculate_packing(items, [bin_data] * max_copies, scan_mode=True)
        all_placed = result["summary"]["all_placed"]
        util       = result["summary"]["avg_utilization"]
        num_bins   = result["summary"]["total_bins_used"]

        if all_placed:
            score = 10000 - num_bins * 10 + util
        else:
            placed_count = sum(b["item_count"] for b in result["packed_bins"])
            score = placed_count * 10 + util

        if score > best_score:
            best_score      = score
            best_bin_data   = bin_data
            best_bin_result = {
                "bin_type":        bin_data["type"],
                "utilization":     util,
                "all_placed":      all_placed,
                "total_bins_used": num_bins,
                "unplaced":        result["unplaced_items"],
            }

        # 贪心已找到单箱且利用率达标，无需继续扫描更大的箱型
        if all_placed and num_bins == 1 and util >= 0.80:
            break

    # 对最优箱型跑完整 OR-Tools，得到精确坐标（用于 3D 展示和费率计算）
    if best_bin_data:
        best_full = calculate_packing(items, [best_bin_data] * max_copies)

    best_util  = best_bin_result["utilization"]     if best_bin_result else 0
    best_all   = best_bin_result["all_placed"]      if best_bin_result else False
    best_type  = best_bin_result["bin_type"]        if best_bin_result else "无"
    print(f"[recommend] 扫描结果: best={best_type}, util={best_util:.2%}, all_placed={best_all}", flush=True)
    best_bins  = best_bin_result["total_bins_used"] if best_bin_result else 0

    # 计算现有最优箱型的运费档级
    existing_fee = None
    existing_tier_rank = len(TIER_ORDER)
    if best_bin_data:
        try:
            existing_fee = calc_bin_fee(best_bin_data, total_weight_kg, sale_price_usd, product_category)
            existing_tier_rank = tier_rank(existing_fee["tier"])
        except Exception:
            pass

    # ── 第二步：推荐新包材，计算运费档级 ─────────────────────────────────────────
    rec_bin    = _calc_recommended_bin(items)
    rec_result = calculate_packing(items, [rec_bin] * max_copies)

    rec_util      = rec_result["summary"]["avg_utilization"]
    rec_placed    = rec_result["summary"]["all_placed"]
    rec_bins_used = rec_result["summary"]["total_bins_used"]

    # ── 第2.2步：用实际装箱 bbox 收紧推荐尺寸 ────────────────────────────────────
    # _calc_recommended_bin 按叠放方式估算，但实际算法可能把货物平铺展开，
    # 导致箱子某维度虚大。直接用装箱后货物的实际 bbox + buffer 更新尺寸，
    # 无需重新跑装箱（货物已在 bbox 内，物理合法）。
    if rec_placed and rec_bins_used == 1 and rec_result["packed_bins"]:
        placed_items = rec_result["packed_bins"][0]["items"]
        if placed_items:
            ax = max(p["position"]["x"] + p["dimensions"]["length"] for p in placed_items)
            ay = max(p["position"]["y"] + p["dimensions"]["width"]  for p in placed_items)
            az = max(p["position"]["z"] + p["dimensions"]["height"] for p in placed_items)
            # 加缓冲后取整，但不超过原尺寸（只收紧不放大）
            new_l = min(math.ceil((ax + BIN_SIZE_BUFFER) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO,
                        rec_bin["length"])
            new_w = min(math.ceil((ay + BIN_SIZE_BUFFER) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO,
                        rec_bin["width"])
            new_h = min(math.ceil((az + BIN_SIZE_BUFFER) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO,
                        rec_bin["height"])
            if (new_l, new_w, new_h) != (rec_bin["length"], rec_bin["width"], rec_bin["height"]):
                orig_l, orig_w, orig_h = rec_bin["length"], rec_bin["width"], rec_bin["height"]
                rec_bin = {**rec_bin,
                           "length": float(new_l), "width": float(new_w), "height": float(new_h)}
                new_vol  = new_l * new_w * new_h
                item_vol = sum(
                    p["dimensions"]["length"] * p["dimensions"]["width"] * p["dimensions"]["height"]
                    for p in placed_items
                )
                new_util = round(item_vol / new_vol, 2) if new_vol > 0 else 0.0
                new_bin_record = {
                    **rec_result["packed_bins"][0],
                    "dimensions": {"length": float(new_l), "width": float(new_w), "height": float(new_h)},
                    "utilization": new_util,
                }
                rec_result = {**rec_result,
                              "packed_bins": [new_bin_record],
                              "summary": {**rec_result["summary"], "avg_utilization": new_util}}
                rec_util = new_util
                logger.info("[尺寸收紧] %.1f×%.1f×%.1f → %.1f×%.1f×%.1f cm, 利用率 %.0f%%",
                            orig_l, orig_w, orig_h, new_l, new_w, new_h, new_util * 100)

    rec_fee = None
    rec_tier_rank = len(TIER_ORDER)
    # 推荐新包材必须单箱装完所有货物，否则视为无效推荐
    rec_single_ok = rec_placed and rec_bins_used == 1
    if rec_single_ok:
        try:
            rec_fee = calc_bin_fee(rec_bin, total_weight_kg, sale_price_usd, product_category)
            rec_tier_rank = tier_rank(rec_fee["tier"])
        except Exception:
            pass

    # ── 第2.5步：若推荐包材未能降档，逐一尝试各档边界包材 ───────────────────────
    # 背景：_calc_recommended_bin 加了 BIN_SIZE_BUFFER 缓冲，当货物尺寸恰好在档位
    # 边界时，缓冲后的推荐尺寸可能越过边界进入更高档，导致无法发现真实的降档机会。
    # 此处从低到高遍历边界包材，只要货物能单箱装完且档级更低，即改用该方案。
    if rec_fee is None or rec_tier_rank >= existing_tier_rank:
        for boundary_tier_name, boundary_box_tmpl in _TIER_BOUNDARY_BOXES:
            b_rank = tier_rank(boundary_tier_name)
            if b_rank >= existing_tier_rank:
                continue  # 只尝试比现有方案档级更低的包材
            boundary_box = dict(boundary_box_tmpl)
            test_result = calculate_packing(items, [boundary_box] * max_copies)
            # 边界包材也必须单箱装完
            if (test_result["summary"]["all_placed"]
                    and test_result["summary"]["total_bins_used"] == 1):
                rec_bin       = boundary_box
                rec_result    = test_result
                rec_util      = rec_result["summary"]["avg_utilization"]
                rec_placed    = True
                rec_bins_used = 1
                rec_single_ok = True
                try:
                    rec_fee       = calc_bin_fee(rec_bin, total_weight_kg, sale_price_usd, product_category)
                    rec_tier_rank = tier_rank(rec_fee["tier"])
                except Exception:
                    pass
                logger.info("[降档检测] 货物可单箱装入 %s 边界包材（%.0f×%.0f×%.0f），档级 %s",
                            boundary_tier_name,
                            boundary_box["length"], boundary_box["width"], boundary_box["height"],
                            rec_fee["tier"] if rec_fee else "未知")
                break  # 找到最低可达档位即停止

    # ── 第2.6步：软包材方案 ──────────────────────────────────────────────────────
    soft_ok = False
    soft_result = None
    soft_bin_data = None
    soft_fee = None
    soft_tier_rank = len(TIER_ORDER)
    if len(items) > 0 and all(i.get("soft_packaging_ok", False) for i in items):
        soft_pack = calc_soft_packing(items)
        if soft_pack:
            soft_ok = True
            soft_result = soft_pack
            soft_bin_data = soft_pack["bin_data"]
            try:
                soft_fee = calc_bin_fee(soft_bin_data, total_weight_kg, sale_price_usd, product_category)
                soft_tier_rank = tier_rank(soft_fee["tier"])
            except Exception:
                pass
    soft_tier_upgrade = (soft_fee is not None and existing_fee is not None
                         and soft_tier_rank < existing_tier_rank)

    # ── 第三步：判断是否因档级更低而强制推荐 ─────────────────────────────────────
    # 推荐包材档级比现有箱型更低，且能单箱装完 → 直接推荐，不考虑利用率
    tier_upgrade = (rec_fee is not None and existing_fee is not None
                    and rec_tier_rank < existing_tier_rank
                    and rec_single_ok)

    # 现有箱型已最优（单箱装完 + 利用率≥80% + 无任何降档机会）时早返回
    if best_all and best_bins == 1 and best_util >= 0.80 and not tier_upgrade and not soft_tier_upgrade:
        compare_summary = {
            "recommended_bin":    None,
            "recommended_result": None,
            "best_existing_bin":    best_type,
            "best_existing_result": {
                "utilization":    best_util,
                "all_placed":     best_all,
                "total_bins_used": best_bins,
                "unplaced":       [],
            },
            "winner":                 best_type,
            "existing_fee":           existing_fee,
            "recommended_fee":        None,
            "tier_upgrade":           False,
            "soft_bin":               soft_bin_data if soft_ok else None,
            "soft_result":            soft_result["packed_bins"][0] if (soft_ok and soft_result) else None,
            "soft_fee":               soft_fee,
            "best_existing_cost_price": best_bin_data.get("cost_price") if best_bin_data else None,
            "best_existing_sku":        best_bin_data.get("sku") if best_bin_data else None,
        }
        full_compare = {
            "recommended_bin":    None,
            "recommended_result": None,
            "best_existing_bin":  best_type,
            "best_full_result":   best_full,
            "soft_full_result":   soft_result if soft_ok else None,
            "compare_summary":    compare_summary,
        }
        print(f"[recommend] 早返回: winner={best_type!r}, best_full={'None' if best_full is None else 'dict'}", flush=True)
        return json.dumps(compare_summary, ensure_ascii=False), full_compare, best_full

    # ── 第四步：综合判断 winner ────────────────────────────────────────────────
    # 收集可行方案，按(档级, 总运费)升序，取最优；无费率信息时退回简单逻辑
    _candidates = []
    if rec_single_ok:
        _candidates.append((rec_tier_rank,      (rec_fee or {}).get("total_fee",      999), "推荐新包材"))
    if soft_ok:
        _candidates.append((soft_tier_rank,     (soft_fee or {}).get("total_fee",     999), "软包材"))
    if best_all:
        _candidates.append((existing_tier_rank, (existing_fee or {}).get("total_fee", 999), best_type))

    if _candidates:
        _candidates.sort()
        winner = _candidates[0][2]
    elif rec_single_ok:
        winner = "推荐新包材"
    elif soft_ok:
        winner = "软包材"
    else:
        winner = best_type

    # 推荐包材无法单箱装完时，不对外暴露推荐数据（防止前端展示多箱推荐）
    if rec_single_ok:
        rec_summary_entry = {
            "utilization":     rec_util,
            "all_placed":      rec_placed,
            "total_bins_used": rec_bins_used,
            "unplaced":        rec_result["unplaced_items"],
        }
        full_rec_result = rec_result
        out_rec_bin     = rec_bin
        out_rec_fee     = rec_fee
    else:
        rec_summary_entry = None
        full_rec_result   = None
        out_rec_bin       = None
        out_rec_fee       = None

    # 软包材输出
    if soft_ok and soft_result:
        soft_summary_entry = {
            "utilization":     soft_result["summary"]["avg_utilization"],
            "all_placed":      True,
            "total_bins_used": 1,
            "unplaced":        [],
        }
        out_soft_bin     = soft_bin_data
        out_soft_fee     = soft_fee
        full_soft_result = soft_result
    else:
        soft_summary_entry = None
        out_soft_bin       = None
        out_soft_fee       = None
        full_soft_result   = None

    compare_summary = {
        "recommended_bin":    out_rec_bin,
        "recommended_result": rec_summary_entry,
        "best_existing_bin":    best_type,
        "best_existing_result": {
            "utilization":    best_util,
            "all_placed":     best_all,
            "total_bins_used": best_bins,
            "unplaced":       best_bin_result["unplaced"] if best_bin_result else [],
        },
        "winner":                   winner,
        "existing_fee":             existing_fee,
        "recommended_fee":          out_rec_fee,
        "tier_upgrade":             tier_upgrade,
        "soft_bin":                 out_soft_bin,
        "soft_result":              soft_summary_entry,
        "soft_fee":                 out_soft_fee,
        "best_existing_cost_price": best_bin_data.get("cost_price") if best_bin_data else None,
        "best_existing_sku":        best_bin_data.get("sku") if best_bin_data else None,
    }

    full_compare = {
        "recommended_bin":    out_rec_bin,
        "recommended_result": full_rec_result,
        "best_existing_bin":  best_type,
        "best_full_result":   best_full,
        "soft_full_result":   full_soft_result,
        "compare_summary":    compare_summary,
    }

    # primary_result = winner 对应的装箱数据
    if winner == "推荐新包材":
        primary_result = full_rec_result or best_full or rec_result
    elif winner == "软包材":
        primary_result = full_soft_result or best_full
    else:
        primary_result = best_full or rec_result
    print(f"[recommend] winner={winner!r}, primary_result={'None' if primary_result is None else type(primary_result).__name__}, "
          f"full_rec_result={'None' if full_rec_result is None else 'dict'}, "
          f"best_full={'None' if best_full is None else 'dict'}, "
          f"rec_result={'None' if rec_result is None else 'dict'}", flush=True)
    return json.dumps(compare_summary, ensure_ascii=False), full_compare, primary_result


# ── 工具执行 ──────────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict, bins: list = None) -> tuple:
    """
    Returns:
        (agent_str, full_result, compare_result)
        agent_str:      返回给 Agent 的精简摘要
        full_result:    完整装箱坐标数据
        compare_result: 对比数据（仅 recommend_and_compare 时有值）
    """
    available = bins if bins else AVAILABLE_BINS
    if tool_name == "get_available_bins":
        hard_count = sum(1 for c in _RAW_CATALOG if c.get("type") == "硬包材")
        soft_count = sum(1 for c in _RAW_CATALOG if c.get("type") == "软包材")
        summary = {
            "message": (
                f"包材目录已加载，共 {len(_RAW_CATALOG)} 个包材（硬包材 {hard_count}，软包材 {soft_count}）。"
                "系统将在 recommend_and_compare 时自动扫描目录，找出最优硬包材，无需手动选择。"
                "请直接调用 recommend_and_compare 工具完成推荐对比。"
            ),
            "total": len(_RAW_CATALOG),
            "hard_bins": hard_count,
            "soft_bins": soft_count,
        }
        return json.dumps(summary, ensure_ascii=False), None, None

    if tool_name == "calculate_packing":
        full = calculate_packing(tool_input["items"], tool_input["bins"])
        summary = {
            "total_bins_used": full["summary"]["total_bins_used"],
            "avg_utilization": full["summary"]["avg_utilization"],
            "all_placed":      full["summary"]["all_placed"],
            "unplaced_items":  full["unplaced_items"],
            "bins": [
                {
                    "bin_type":     b["bin_type"],
                    "utilization":  b["utilization"],
                    "item_count":   b["item_count"],
                    "total_weight": b["total_weight"],
                }
                for b in full["packed_bins"]
            ],
        }
        return json.dumps(summary, ensure_ascii=False), full, None

    if tool_name == "recommend_and_compare":
        agent_str, full_compare, primary_result = _do_recommend_and_compare(tool_input["items"], bins=bins)
        winner_in_compare = (full_compare or {}).get("compare_summary", {}).get("winner", "?")
        print(f"[execute_tool] primary_result={'None' if primary_result is None else type(primary_result).__name__}, "
              f"full_compare={'None' if full_compare is None else 'dict'}, winner={winner_in_compare}", flush=True)
        return agent_str, primary_result, full_compare

    return json.dumps({"error": f"未知工具: {tool_name}"}), None, None


# ── 带重试的 API 调用（处理 429 限流）────────────────────────────────────────

def _call_api_simple(messages: list, max_retries: int = 3) -> object:
    """不带 tools 的普通对话调用，用于生成最终说明文字"""
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=messages,
            )
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


# ── 本地兜底摘要生成 ──────────────────────────────────────────────────────────

def _local_summary(items: list, final_result: dict, compare_result: dict | None) -> str:
    """当 Claude API 不可用时，用 Python 本地生成摘要文字。"""
    util_pct   = final_result["summary"]["avg_utilization"] * 100
    bins_used  = final_result["summary"]["total_bins_used"]
    all_placed = final_result["summary"]["all_placed"]
    bin_types  = "、".join(dict.fromkeys(b["bin_type"] for b in final_result["packed_bins"]))

    if all_placed:
        text = (
            f"本次共 {len(items)} 件货物，全部装入 {bins_used} 个箱子（{bin_types}），"
            f"平均空间利用率 {util_pct:.0f}%。"
        )
    else:
        unplaced_count = len(final_result["unplaced_items"])
        text = (
            f"本次共 {len(items)} 件货物，有 {unplaced_count} 件无法装入，"
            f"已装入 {bins_used} 个箱子（{bin_types}），平均利用率 {util_pct:.0f}%。"
        )

    if compare_result:
        s = compare_result["compare_summary"]
        ef = s.get("existing_fee") or {}
        rf = s.get("recommended_fee") or {}

        def fee_str(f: dict) -> str:
            if not f:
                return ""
            return (f"FBA费档级：{f.get('tier','')}，"
                    f"预估总FBA费：${f.get('total_fee', 0):.2f}"
                    f"（FBA费${f.get('shipping_fee',0):.2f}"
                    f"+包装费${f.get('packaging_fee',0):.2f}"
                    f"+附加费${f.get('surcharge',0):.2f}）")

        sf = s.get("soft_fee") or {}
        soft_b = s.get("soft_bin")
        soft_r = s.get("soft_result")

        if s.get("recommended_bin") or soft_b:
            tier_tip = ""
            if s.get("tier_upgrade"):
                tier_tip = f"【FBA费降档：{ef.get('tier','')} → {rf.get('tier','')}，强制推荐】"
            parts = []
            if s.get("recommended_bin"):
                rec_b = s["recommended_bin"]
                parts.append(
                    f"推荐新包材 {rec_b['length']}×{rec_b['width']}×{rec_b['height']}cm"
                    f"（利用率{s['recommended_result']['utilization']*100:.0f}%）"
                )
            if soft_b and soft_r:
                parts.append(
                    f"软包材 {soft_b['length']}×{soft_b['width']}×{soft_b['height']}cm"
                    f"（利用率{soft_r['utilization']*100:.0f}%）"
                )
            parts.append(f"{s['best_existing_bin']}（利用率{s['best_existing_result']['utilization']*100:.0f}%）")
            text += f"可选方案：{'、'.join(parts)}，建议选用：{s['winner']}。{tier_tip}"
            if rf:
                text += f" 推荐新包材{fee_str(rf)}。"
            if sf:
                text += f" 软包材{fee_str(sf)}。"
            if ef:
                text += f" 现有箱型{fee_str(ef)}。"
        else:
            text += (
                f"现有箱型「{s['best_existing_bin']}」利用率已达"
                f"{s['best_existing_result']['utilization']*100:.0f}%，无需定制新包材。"
            )
            if ef:
                text += f" {fee_str(ef)}。"
    return text


# ── Agent 主函数 ──────────────────────────────────────────────────────────────

def run_packing_agent(items: list, bins: list = None) -> dict:
    """
    AI Agent 模式：Claude 调用 recommend_and_compare 工具完成装箱决策。
      1. 发送货物清单给 Claude，仅提供 recommend_and_compare 工具
      2. Claude 调用该工具，系统自动扫描包材目录并返回结果
      3. Claude 生成中文总结
      4. API 不可用时自动降级为本地 Python 逻辑 + 本地摘要
    """
    available = bins if bins else (_prefilter_catalog_bins(items) if _RAW_CATALOG else AVAILABLE_BINS)
    print(f"[agent] run_packing_agent: {len(items)}件货物, available={len(available)}个包材, bins_param={'自定义' if bins else '目录扫描'}", flush=True)

    system_prompt = (
        "你是三维装箱专家AI助手。你只有一个工具：recommend_and_compare。\n"
        "请立即调用该工具，传入完整货物列表。工具会自动从包材目录扫描最优方案。\n"
        "工具返回结果后，给出简洁的中文总结：推荐包材、利用率、FBA费档级结论。"
    )
    user_message = (
        f"请为以下 {len(items)} 件货物制定最优装箱方案：\n"
        + json.dumps(items, ensure_ascii=False)
    )

    messages      = [{"role": "user", "content": user_message}]
    packing_results: list = []
    compare_result        = None
    final_result          = None
    agent_summary         = None
    ai_used               = False   # 标记本次是否真正经过 Claude 决策
    ai_error              = None    # 记录 API 失败原因

    try:
        for turn in range(10):          # 最多 10 轮，防止无限循环
            # 还没拿到计算结果前强制调用工具，防止 Claude 只输出文字就结束
            tool_choice = {"type": "any"} if final_result is None else {"type": "auto"}
            logger.info("[Agent 第%d轮] tool_choice=%s", turn + 1, tool_choice["type"])

            response = client.messages.create(
                model       = "claude-haiku-4-5-20251001",
                max_tokens  = 4096,
                system      = system_prompt,
                tools       = TOOLS,
                tool_choice = tool_choice,
                messages    = messages,
            )

            # ── 将 assistant 回复追加到历史 ──
            messages.append({"role": "assistant", "content": response.content})

            # ── 收集本轮的文字块与工具调用块 ──
            tool_uses   = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            logger.info("[Agent 第%d轮] stop_reason=%s tool_calls=%s",
                        turn + 1, response.stop_reason,
                        [tu.name for tu in tool_uses])

            if response.stop_reason == "end_turn":
                if text_blocks:
                    agent_summary = text_blocks[-1].text
                    logger.info("[Agent 第%d轮] end_turn，summary=%s", turn + 1, agent_summary[:80])
                break

            if response.stop_reason == "tool_use" and tool_uses:
                tool_results = []
                for tu in tool_uses:
                    logger.info("[Agent 第%d轮] 执行工具: %s input_keys=%s",
                                turn + 1, tu.name, list(tu.input.keys()))
                    result_str, full, cr = execute_tool(tu.name, tu.input, bins=available)
                    logger.info("[Agent 第%d轮] 工具结果 full_is_none=%s", turn + 1, full is None)
                    if full is not None:
                        packing_results.append(full)
                        final_result = full
                        ai_used = True      # Claude 成功调用了计算工具
                    if cr is not None:
                        compare_result = cr
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": tu.id,
                        "content":     result_str,
                    })
                messages.append({"role": "user", "content": tool_results})
            else:
                # 意外的 stop_reason，直接退出
                logger.warning("[Agent 第%d轮] 意外 stop_reason=%s，退出循环", turn + 1, response.stop_reason)
                if text_blocks:
                    agent_summary = text_blocks[-1].text
                break

    except Exception as e:
        # API 不可用（IP 被封、网络超时等），降级为本地逻辑
        ai_error = f"{type(e).__name__}: {e}"
        logger.error("Claude API 调用失败，降级为本地计算: %s", ai_error)

    # ── 兜底：若 Claude 没有执行计算（API 不可用），本地直接跑推荐逻辑 ──────────
    if final_result is None or compare_result is None:
        _, full_compare, rec_result = _do_recommend_and_compare(items, bins=bins)
        compare_result = full_compare
        if rec_result is not None:
            if rec_result not in packing_results:
                packing_results.append(rec_result)
            final_result = rec_result
        elif final_result is None and full_compare:
            # 从 compare_result 里取 best_full
            bf = full_compare.get("best_full_result")
            if bf:
                packing_results.append(bf)
                final_result = bf

    # ── 兜底：若没有 AI 生成摘要，本地生成 ───────────────────────────────────
    if not agent_summary:
        agent_summary = _local_summary(items, final_result, compare_result)

    fr_bin = (final_result or {}).get("packed_bins", [{}])[0].get("bin_type", "无") if final_result else "无"
    print(f"[agent] 最终结果: final_bin={fr_bin}, ai_used={ai_used}, ai_error={ai_error}", flush=True)
    return {
        "success":         True,
        "agent_summary":   agent_summary,
        "ai_used":         ai_used,
        "ai_error":        ai_error,
        "packing_results": packing_results,
        "final_result":    final_result,
        "compare_result":  compare_result,
    }
