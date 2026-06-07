"""
三维装箱 AI Agent
使用 Claude 作为决策大脑，自动选择箱型、调用算法、验证方案。
当箱型库没有合适箱型时，自动推荐新包材并与最优现有箱型对比。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import json
import math
import time
import logging
import os as _os
import httpx
import anthropic
import openai as _openai
import pymysql as _pymysql
import pymysql.cursors as _pymysql_cursors

logger = logging.getLogger(__name__)
from packing_engine import calculate_packing, AVAILABLE_BINS, MAX_BIN_WEIGHT
from config import (MAX_BIN_LONG_SIDE, MAX_BIN_SHORT_SIDE, BIN_SIZE_BUFFER, BIN_SIZE_ROUND_TO,
                    MIN_UTILIZATION_THRESHOLD,
                    SOFT_PKG_HEIGHT_FACTOR, SOFT_PKG_SIDE_MARGIN, SOFT_PKG_MAX_WEIGHT)
from shipping_tiers import calc_bin_fee, tier_rank, TIER_ORDER
from soft_packing_engine import calc_soft_packing

client = anthropic.Anthropic()

# ── DeepSeek / OpenAI-compatible 适配层 ───────────────────────────────────────

class _Block:
    """伪造 Claude 风格的响应块，兼容 text / tool_use 两种类型"""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

class _Usage:
    def __init__(self, inp, out):
        self.input_tokens  = inp
        self.output_tokens = out

class _Resp:
    def __init__(self, content, stop_reason, usage):
        self.content     = content
        self.stop_reason = stop_reason
        self.usage       = usage

def _get_ds_client():
    return _openai.OpenAI(
        api_key  = _os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url = "https://api.deepseek.com",
    )

def _get_qwen_client():
    return _openai.OpenAI(
        api_key  = _os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def _to_oai_tools(tools: list) -> list:
    """Claude tools → OpenAI function-calling 格式"""
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters":  t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]

def _to_oai_messages(messages: list, system: str = None) -> list:
    """Claude-format messages → OpenAI-format messages（含 system 注入）"""
    result = []
    if system:
        result.append({"role": "system", "content": system})
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        # 推理模型原始 OAI assistant 消息，直接透传（保留 reasoning_content）
        if role == "assistant" and "_raw_oai_msg" in msg:
            raw = msg["_raw_oai_msg"]
            m = {"role": "assistant", "content": raw.content or ""}
            if getattr(raw, "reasoning_content", None):
                m["reasoning_content"] = raw.reasoning_content
            if getattr(raw, "tool_calls", None):
                m["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in raw.tool_calls
                ]
            result.append(m)
            continue
        # 纯文本消息
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            # tool_result → 独立 tool 消息
            if role == "user":
                tool_results = [b for b in content
                                if (isinstance(b, dict) and b.get("type") == "tool_result")]
                if tool_results:
                    for b in tool_results:
                        result.append({
                            "role":         "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content":      b["content"],
                        })
                    continue
            # assistant 消息（可能含 tool_use 块）
            if role == "assistant":
                texts, tool_calls = [], []
                for b in content:
                    btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                    if btype == "text":
                        texts.append(b["text"] if isinstance(b, dict) else b.text)
                    elif btype == "tool_use":
                        bid   = b["id"]    if isinstance(b, dict) else b.id
                        bname = b["name"]  if isinstance(b, dict) else b.name
                        binp  = b["input"] if isinstance(b, dict) else b.input
                        tool_calls.append({
                            "id":   bid,
                            "type": "function",
                            "function": {
                                "name":      bname,
                                "arguments": json.dumps(binp, ensure_ascii=False),
                            },
                        })
                m = {"role": "assistant", "content": " ".join(texts) or None}
                if tool_calls:
                    m["tool_calls"] = tool_calls
                result.append(m)
                continue
        result.append({"role": role, "content": str(content)})
    return result

def _from_oai_resp(resp) -> _Resp:
    """OpenAI response → Claude-compatible 伪响应"""
    choice = resp.choices[0]
    msg    = choice.message
    blocks = []
    if msg.content:
        blocks.append(_Block(type="text", text=msg.content))
    if msg.tool_calls:
        for tc in msg.tool_calls:
            blocks.append(_Block(
                type  = "tool_use",
                id    = tc.id,
                name  = tc.function.name,
                input = json.loads(tc.function.arguments),
            ))
    stop  = "end_turn" if choice.finish_reason == "stop" else "tool_use"
    usage = _Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
    r = _Resp(blocks, stop, usage)
    # 保存原始 OpenAI message，推理模型多轮时需要把 reasoning_content 带回
    r._raw_oai_msg = msg
    return r

# ── 包材目录（从 MySQL bins_catalog 表加载）─────────────────────────────────
_CATALOG_DB_CONFIG = {
    "host":        "127.0.0.1",
    "port":        3306,
    "user":        "root",
    "password":    "Deng123456*",
    "database":    "packing_demo",
    "charset":     "utf8mb4",
    "cursorclass": _pymysql_cursors.DictCursor,
}

def _load_catalog_from_db() -> list:
    """查询 bins_catalog 表，返回与原 JSON 格式兼容的 dict 列表。"""
    try:
        conn = _pymysql.connect(**_CATALOG_DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sku, name, price, length, width, height, mat_weight, "
                "       bin_type, mat_type, protection_level, protection_rank, max_weight "
                "FROM bins_catalog ORDER BY bin_type, mat_type, length"
            )
            rows = cur.fetchall()
        conn.close()
        result = []
        for r in rows:
            result.append({
                "sku":              r["sku"],
                "name":             r["name"],
                "price":            float(r["price"] or 0),
                "length":           float(r["length"]),
                "width":            float(r["width"]),
                "height":           float(r["height"]),
                "mat_weight":       float(r.get("mat_weight") or 0),
                "type":             r["bin_type"],       # 硬包材 / 软包材
                "mat_name":         r["mat_type"],       # 三层纸箱 / 纸箱 / 单层纸箱 / 袋子
                "protection_level": r["protection_level"],
                "protection_rank":  int(r["protection_rank"]),
                "max_weight":       float(r.get("max_weight") or 22),
            })
        return result
    except Exception as e:
        logger.warning("[catalog] DB加载失败，使用空目录: %s", e)
        return []

_RAW_CATALOG: list = _load_catalog_from_db()
print(f"[agent] 包材目录已从DB加载：共 {len(_RAW_CATALOG)} 条", flush=True)


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


def _catalog_to_soft_bin(c: dict, stack_h: float) -> dict:
    """软包材目录条目 → bin_data dict（供费率计算和结果展示使用）。
    对 height=0 的平面袋，用货物叠放高度 stack_h 作为有效厚度。"""
    mat = (c.get("mat_name") or "").strip()
    name_short = c["name"][:20]
    display_name = mat or name_short
    sku = c.get("sku", "")
    eff_h = float(c.get("height", 0))
    if eff_h <= 0:
        eff_h = round(stack_h, 1)
    return {
        "type":       f"{display_name} ({sku})" if sku else display_name,
        "length":     float(c["length"]),
        "width":      float(c["width"]),
        "height":     eff_h,
        "max_weight": float(c.get("max_weight", 22)),
        "cost_price": float(c.get("price", 0)),
        "sku":        sku,
    }


# 类型关键词同义词扩展表（方向二）
# 查询 key 时，同时接受 value 列表中的任意词作为匹配
_TYPE_SYNONYMS: dict = {
    "三层纸箱": ["三层纸箱"],          # 精确匹配 mat_type=三层纸箱
    "纸箱":    ["纸箱"],               # 精确匹配 mat_type=纸箱
    "单层纸箱": ["单层纸箱"],          # 精确匹配 mat_type=单层纸箱
    "袋子":    ["袋子", "袋"],
    # 兼容旧名称（自定义包材/重推时可能传入）
    "纸盒":    ["纸盒", "纸箱", "飞机盒", "瓦楞纸盒"],
    "飞机盒":  ["飞机盒", "纸箱", "纸盒"],
}

import re as _re

def _type_matches(c: dict, preferred_type: str) -> bool:
    """
    preferred_type 过滤（方向一 + 方向二）：
    - mat_name 有值时：只匹配 mat_name，避免描述文字误匹配
    - mat_name 为空时：取 name 首段（括号/逗号/空格前）做主类型匹配
    - 两路均使用同义词扩展表（_TYPE_SYNONYMS）
    """
    keywords = _TYPE_SYNONYMS.get(preferred_type, [preferred_type])
    mat_name = (c.get("mat_name") or "").strip()
    if mat_name:
        return any(kw in mat_name for kw in keywords)
    # mat_name 为空：取 name 首段作为主类型
    full_name = c.get("name") or ""
    primary = _re.split(r'[（(，,\s]', full_name)[0].strip()
    return any(kw in primary for kw in keywords)


def _prefilter_catalog_bins(items: list, max_results: int = 150,
                             excluded_skus: set = None, max_cost: float = None,
                             preferred_type: str = None, require_tight: bool = False,
                             min_protection_rank: int = 1) -> list:
    """
    从目录中筛选适合当前货物的硬包材候选箱型，按体积升序排列。
    筛选条件：
      1. 箱型承重 >= 货物总重
      2. 箱型体积 >= 货物总体积
      3. 每件货物（旋转后）均能放入箱中（三边升序逐一对比）
    重推约束（可选）：
      excluded_skus       — 排除指定 SKU（如当前推荐不满意）
      max_cost            — 成本上限（元）
      preferred_type      — 偏好类型关键词（如"纸箱"/"袋"），支持同义词扩展
      require_tight       — 紧凑匹配：箱体积不超过货物总体积的 3 倍
      min_protection_rank — 最低防护等级(1-4)，由产品防护级别推算
    """
    if not _RAW_CATALOG:
        return AVAILABLE_BINS
    total_weight     = sum(i.get("weight", 0) for i in items)
    total_vol        = sum(i["length"] * i["width"] * i["height"] for i in items)
    max_item_vol     = max(i["length"] * i["width"] * i["height"] for i in items)
    max_item_weight  = max(i.get("weight", 0) for i in items)
    item_sorted_dims = [sorted([i["length"], i["width"], i["height"]]) for i in items]
    excluded_skus    = excluded_skus or set()

    candidates = []
    for c in _RAW_CATALOG:
        if c.get("type") != "硬包材":
            continue
        if c.get("mat_name") == "定制包材":
            continue
        if c.get("sku") in excluded_skus:
            continue
        if c.get("protection_rank", 1) < min_protection_rank:
            continue
        # 承重：至少能装下最重的单件（多箱时不要求承受全部总重）
        if c.get("max_weight", 22) < max_item_weight:
            continue
        bin_vol = c["length"] * c["width"] * c["height"]
        # 体积：至少能装下体积最大的单件（支持多箱，不再要求 >= 总体积）
        if bin_vol < max_item_vol:
            continue
        if require_tight and bin_vol > total_vol * 3:
            continue
        if max_cost is not None and float(c.get("price", 0)) > max_cost:
            continue
        if preferred_type and not _type_matches(c, preferred_type):
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


def _prefilter_soft_catalog_bins(items: list, max_results: int = 8,
                                   min_protection_rank: int = 1,
                                   height_factor: float = None,
                                   side_margin: float = None) -> list:
    """
    从软包材目录中筛选可容纳货物的袋子候选（排除气泡袋辅材）。
    匹配逻辑：袋子开口尺寸 >= 货物软包装所需展开尺寸（用软包材公式计算）。
    返回原始目录条目列表，按价格升序排列。
    height_factor / side_margin：覆盖 config 全局值，由 AI 决策传入。
    """
    if not _RAW_CATALOG:
        return []
    _hf = height_factor if height_factor is not None else SOFT_PKG_HEIGHT_FACTOR
    _sm = side_margin   if side_margin   is not None else SOFT_PKG_SIDE_MARGIN
    total_weight = sum(i.get("weight", 0) for i in items)

    def _flat(item):
        d = sorted([item["length"], item["width"], item["height"]], reverse=True)
        return d[0], d[1], d[2]

    stack_l = max(_flat(i)[0] for i in items)
    stack_w = max(_flat(i)[1] for i in items)
    stack_h = sum(_flat(i)[2] for i in items)

    req_l = stack_l + stack_h * _hf
    req_w = stack_w + _sm * 2
    req_sorted = sorted([req_l, req_w], reverse=True)

    _EXCL_MAT = {"气泡袋"}
    candidates = []
    for c in _RAW_CATALOG:
        if c.get("type") != "软包材":
            continue
        if (c.get("mat_name") or "").strip() in _EXCL_MAT:
            continue
        if float(c.get("max_weight", 22)) < total_weight:
            continue
        if c.get("protection_rank", 1) < min_protection_rank:
            continue
        bag_dims = sorted([float(c["length"]), float(c["width"])], reverse=True)
        if bag_dims[0] >= req_sorted[0] and bag_dims[1] >= req_sorted[1]:
            candidates.append(c)

    candidates.sort(key=lambda c: (float(c.get("price", 999)),
                                   float(c["length"]) * float(c["width"])))
    logger.info("[软包材筛选] 货物需求展开尺寸%.1f×%.1f cm，匹配到 %d 个候选袋子",
                req_l, req_w, len(candidates))
    return candidates[:max_results]


def _make_soft_catalog_result(items: list, bin_data: dict) -> dict:
    """为软包材目录箱型生成装箱结果结构，格式与 calc_soft_packing() 相同。"""
    def _flat(item):
        d = sorted([item["length"], item["width"], item["height"]], reverse=True)
        return d[0], d[1], d[2]

    total_weight = sum(i.get("weight", 0) for i in items)
    sorted_items = sorted(items, key=lambda x: _flat(x)[0] * _flat(x)[1], reverse=True)

    placed_items = []
    z_offset = 0.0
    for item in sorted_items:
        il, iw, ih = _flat(item)
        placed_items.append({
            "id": item["id"],
            "position": {"x": 0.0, "y": 0.0, "z": z_offset},
            "dimensions": {"length": float(il), "width": float(iw), "height": float(ih)},
            "rotation_type": 0,
        })
        z_offset += ih

    b_l, b_w, b_h = bin_data["length"], bin_data["width"], bin_data["height"]
    bin_vol  = b_l * b_w * b_h
    item_vol = sum(i["length"] * i["width"] * i["height"] for i in items)
    util = round(item_vol / bin_vol, 2) if bin_vol > 0 else 0.0

    return {
        "bin_data": bin_data,
        "packed_bins": [{
            "bin_type":     bin_data["type"],
            "dimensions":   {"length": b_l, "width": b_w, "height": b_h},
            "utilization":  util,
            "total_weight": round(total_weight, 3),
            "item_count":   len(items),
            "items":        placed_items,
        }],
        "unplaced_items": [],
        "summary": {
            "total_bins_used": 1,
            "avg_utilization": util,
            "all_placed":      True,
        },
    }


def _select_soft_bin_ai(items: list, candidates: list,
                         ai_provider: str = "anthropic", ai_model: str = None) -> dict:
    """
    从软包材候选中选出最适合当前产品的袋子。
    候选已按价格升序排列，且均已通过尺寸适配过滤，直接取最便宜的（candidates[0]）。
    注：此前版本在候选 >2 时调用 AI 选型，但 AI 选结果与直接取最便宜几乎一致，
    且每次额外消耗 3-4 秒和数百 Token，已移除 AI 调用，改为纯本地逻辑。
    """
    if not candidates:
        return {}
    logger.info("[软包材选型] %d个候选→取最便宜: %s", len(candidates), candidates[0].get("sku", ""))
    return candidates[0]  # 已按价格升序，第0个最便宜且尺寸适配


# ── AI 参数决策函数 ───────────────────────────────────────────────────────────
# 以下三个函数均遵循"AI失败立即降级"原则：
#   - 只发简短 prompt，max_tokens 极小（8-20 个），耗时 <1s
#   - try/except 全覆盖，任何异常均回退到当前算法逻辑
#   - 参数范围做 clamp，防止 AI 输出极端值破坏后续计算

def _ai_decide_bin_buffer(items: list, classify_result: dict = None,
                           ai_provider: str = "anthropic", ai_model: str = None) -> float:
    """
    方案二：AI 根据产品信息决定定制硬包材的尺寸余量（cm，单边）。
    普通商品 ~1cm，高价值/易碎 3-5cm，衣物/软性 0.5cm。
    失败时降级返回 config.BIN_SIZE_BUFFER（0.5cm）。
    """
    if not classify_result:
        return BIN_SIZE_BUFFER
    protection   = classify_result.get("protection_level", "medium")
    reason       = classify_result.get("reason", "")
    product_name = next((i.get("product_name", "") for i in items if i.get("product_name")), "")
    max_size     = max(max(i["length"], i["width"], i["height"]) for i in items)
    total_weight = sum(i.get("weight", 0) for i in items)
    sale_price   = max((i.get("sale_price", 0) for i in items), default=0)

    prompt = (
        f"包装工程师视角：请为以下产品决定定制纸箱三边各需要留多少尺寸余量（cm）。\n"
        f"产品：{product_name}  防护级别：{protection}（{reason}）\n"
        f"最大单边：{max_size:.1f}cm  总重：{total_weight:.2f}kg  售价：{sale_price:.2f}USD\n"
        f"参考：普通商品1.0、玻璃/陶瓷/精密仪器3.0-5.0、衣物/软件0.5\n"
        f"只输出一个数字（范围0.5-8.0），不要任何解释。"
    )
    # 使用用户选定的模型，未指定时降级到默认快速模型
    _fast_model = (ai_model or "claude-haiku-4-5-20251001") if ai_provider == "anthropic" \
                  else (ai_model or ("qwen3.6-plus" if ai_provider == "qwen" else "deepseek-v4-flash"))
    try:
        resp = _api_with_retry(
            _provider=ai_provider, model=_fast_model, max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        m = _re.search(r'\d+\.?\d*', resp.content[0].text.strip())
        if m:
            val = max(0.5, min(8.0, float(m.group())))
            logger.info("[AI余量] buffer=%.1fcm 产品=%s 防护=%s", val, product_name[:20], protection)
            return val
    except Exception as e:
        logger.warning("[AI余量] 调用失败，降级使用默认值 %.1fcm: %s", BIN_SIZE_BUFFER, e)
    return BIN_SIZE_BUFFER


def _ai_decide_soft_params(items: list, classify_result: dict = None,
                            ai_provider: str = "anthropic", ai_model: str = None,
                            _meta: dict = None) -> tuple:
    """
    方案二（软包材）：AI 决定袋子尺寸参数。
    返回 (height_factor, side_margin)。
      height_factor: 封口折叠比例（0.3-0.7）
      side_margin:   两侧余量（cm，0.1-2.0）
    失败时降级返回 config 全局值。
    _meta: 可选 dict，AI 调用成功时更新 {"used":True, "in_tokens":n, "out_tokens":n}
    """
    if not classify_result:
        return SOFT_PKG_HEIGHT_FACTOR, SOFT_PKG_SIDE_MARGIN
    protection   = classify_result.get("protection_level", "medium")
    reason       = classify_result.get("reason", "")
    product_name = next((i.get("product_name", "") for i in items if i.get("product_name")), "")
    max_h        = max(sorted([i["length"], i["width"], i["height"]])[0] for i in items)

    prompt = (
        f"包装工程师视角：为以下产品决定软包装（袋子）参数。\n"
        f"产品：{product_name}  防护：{protection}（{reason}）  最大高度：{max_h:.1f}cm\n"
        f"输出两个数字，用逗号分隔：\n"
        f"①封口折叠比例(0.3-0.7，普通0.5，厚重0.6，扁平0.4)\n"
        f"②两侧余量cm(0.1-2.0，普通0.2，液体/异形1.0)\n"
        f"示例：0.5,0.2  只输出数字，不要解释。"
    )
    # 使用用户选定的模型，未指定时降级到默认快速模型
    _fast_model = (ai_model or "claude-haiku-4-5-20251001") if ai_provider == "anthropic" \
                  else (ai_model or ("qwen3.6-plus" if ai_provider == "qwen" else "deepseek-v4-flash"))
    try:
        resp = _api_with_retry(
            _provider=ai_provider, model=_fast_model, max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        nums = _re.findall(r'\d+\.?\d*', resp.content[0].text.strip())
        if len(nums) >= 2:
            hf = max(0.3, min(0.7, float(nums[0])))
            sm = max(0.1, min(2.0, float(nums[1])))
            logger.info("[AI软包材参数] factor=%.2f margin=%.2f 产品=%s", hf, sm, product_name[:20])
            if _meta is not None:
                _meta["used"] = True
                _meta["model"]      = _fast_model
                _meta["in_tokens"]  += getattr(resp.usage, "input_tokens",  0) or 0
                _meta["out_tokens"] += getattr(resp.usage, "output_tokens", 0) or 0
            return hf, sm
    except Exception as e:
        logger.warning("[AI软包材参数] 调用失败，降级: %s", e)
    return SOFT_PKG_HEIGHT_FACTOR, SOFT_PKG_SIDE_MARGIN


def _ai_select_winner(items: list, candidates: list, classify_result: dict = None,
                       ai_provider: str = "anthropic", ai_model: str = None,
                       _meta: dict = None) -> str:
    """
    方案一：AI 从多个包材方案中裁决最优 winner，替代原本的纯 sort() 逻辑。
    candidates 格式：[{"name": str, "tier": str, "total_fee": float,
                        "utilization": float, "cost_price": float}, ...]
    失败时降级为按(tier_rank, total_fee)升序取第一。
    _meta: 可选 dict，AI 调用成功时更新 {"used":True, "in_tokens":n, "out_tokens":n}
    """
    # 只有一个候选时无需 AI
    if not candidates:
        return "推荐新包材"
    if len(candidates) == 1:
        return candidates[0]["name"]

    protection   = (classify_result or {}).get("protection_level", "medium")
    reason       = (classify_result or {}).get("reason", "")
    product_name = next((i.get("product_name", "") for i in items if i.get("product_name")), "")
    sale_price   = max((i.get("sale_price", 0) for i in items), default=0)

    # 前置硬规则：若某候选在所有关键指标均劣于另一候选，直接排除，不交给 AI 裁决
    # （防止 AI 因候选名称偏见而选出明显更差的方案）
    def _dominates(a: dict, b: dict) -> bool:
        """a 在费档/总费用/利用率全部 ≥ b（且至少一项严格优），则 a 支配 b"""
        a_tier = tier_rank(a.get("tier", ""))
        b_tier = tier_rank(b.get("tier", ""))
        a_fee  = a.get("total_fee", 999)
        b_fee  = b.get("total_fee", 999)
        a_util = a.get("utilization", 0)
        b_util = b.get("utilization", 0)
        # tier 越小越好，fee 越小越好，util 越大越好
        not_worse  = (a_tier <= b_tier) and (a_fee <= b_fee) and (a_util >= b_util)
        strictly_better = (a_tier < b_tier) or (a_fee < b_fee - 0.01) or (a_util > b_util + 0.01)
        return not_worse and strictly_better

    filtered = list(candidates)
    for cand in list(filtered):
        if any(_dominates(other, cand) for other in filtered if other is not cand):
            logger.info("[winner前置过滤] 排除被支配方案: %s", cand["name"])
            filtered.remove(cand)
    if not filtered:
        filtered = candidates   # 全部互相支配时退回全量

    if len(filtered) == 1:
        logger.info("[winner前置过滤] 只剩一个候选，直接选: %s", filtered[0]["name"])
        return filtered[0]["name"]

    cand_lines = "\n".join(
        f"{i+1}. {c['name']}  FBA费档:{c.get('tier','未知')}  "
        f"总运费:{c.get('total_fee', 0):.2f}元  空间利用率:{c.get('utilization', 0):.0%}  "
        f"包材成本:{c.get('cost_price', 0):.2f}元"
        for i, c in enumerate(filtered)
    )
    prompt = (
        f"包装选型专家：请为以下产品从候选方案中选出最优包材。\n"
        f"产品：{product_name}  防护需求：{protection}（{reason}）  售价：{sale_price:.2f}USD\n\n"
        f"候选方案：\n{cand_lines}\n\n"
        f"原则：①FBA费档低优先（运费更省）②同档时总运费低优先③高价值/易碎品同等条件选利用率高的（减晃动）\n"
        f"只输出选择的序号（纯数字1-{len(filtered)}），不要任何解释。"
    )
    candidates = filtered   # AI 只看过滤后的候选
    # 使用用户选定的模型，未指定时降级到默认快速模型
    _fast_model = (ai_model or "claude-haiku-4-5-20251001") if ai_provider == "anthropic" \
                  else (ai_model or ("qwen3.6-plus" if ai_provider == "qwen" else "deepseek-v4-flash"))
    try:
        resp = _api_with_retry(
            _provider=ai_provider, model=_fast_model, max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        m = _re.search(r'\d+', resp.content[0].text.strip())
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(candidates):
                chosen = candidates[idx]["name"]
                logger.info("[AI winner裁决] %d方案→选%d: %s", len(candidates), idx + 1, chosen)
                if _meta is not None:
                    _meta["used"] = True
                    _meta["model"]      = _fast_model
                    _meta["in_tokens"]  += getattr(resp.usage, "input_tokens",  0) or 0
                    _meta["out_tokens"] += getattr(resp.usage, "output_tokens", 0) or 0
                return chosen
    except Exception as e:
        logger.warning("[AI winner裁决] 调用失败，降级排序: %s", e)

    # 降级：按(tier_rank, total_fee)升序取第一
    sorted_c = sorted(candidates,
                      key=lambda c: (tier_rank(c.get("tier", "")), c.get("total_fee", 999)))
    return sorted_c[0]["name"]


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
            "同时评估软包材方案，输出推荐包材、利用率、FBA费档级对比结果。\n"
            "支持约束参数优化方案：可指定偏好类型、紧凑匹配、排除特定包材、成本上限。\n"
            "若首次结果不满足产品防护需求，可调整参数再次调用（最多共调用2次）。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "货物列表（可省略，系统已有货物数据，无需重复传入）",
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
                "preferred_type": {
                    "type": "string",
                    "description": "偏好包材类型关键词，如'纸箱'/'硬盒'/'袋子'。仅返回名称含此关键词的包材。",
                },
                "require_tight": {
                    "type": "boolean",
                    "description": "是否要求紧凑匹配：箱体积不超过货物总体积3倍。适合高防护/高价值产品，避免推荐虚大包材。",
                },
                "excluded_skus": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "排除的包材SKU列表，重推时跳过已推荐但不满意的方案。",
                },
                "max_cost": {
                    "type": "number",
                    "description": "包材成本上限（元），超出此价格的包材不纳入候选。",
                },
            },
            "required": [],
        },
    },
]

print(f"[agent] TOOLS 已注册：{[t['name'] for t in TOOLS]}", flush=True)

# ── 推荐新包材算法 ────────────────────────────────────────────────────────────

def _calc_recommended_bin(items: list, size_buffer: float = None) -> dict:
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

    # 加缓冲余量，向上取整到指定精度（size_buffer 由 AI 决策传入，缺省用 config 全局值）
    _buf = size_buffer if size_buffer is not None else BIN_SIZE_BUFFER
    rec_l = math.ceil((max_l + _buf) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO
    rec_w = math.ceil((max_w + _buf) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO
    rec_h = math.ceil((max_h + _buf) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO

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


def _do_recommend_soft(items: list, ai_provider: str = "anthropic", ai_model: str = None,
                        min_protection_rank: int = 1, classify_result: dict = None) -> tuple:
    """
    软包材专用推荐流程（当所有货物都设置了 soft_packaging_ok=True 时调用）。

    逻辑与硬包材对称：
      - 目录袋 = "包材库最优"（best_existing）
      - 公式定制袋 = "推荐新包材"（recommended）
      - 判断目录袋是否"足够优"（util≥80% 且费档不劣于定制）：
          · 够优 → 只推荐目录袋，不展示定制方案
          · 不够 → 两者对比，按(费档, 总费用)选 winner

    Returns 与 _do_recommend_and_compare 相同的 tuple 格式。
    """
    _SOFT_GOOD_UTIL = 0.80   # 目录袋利用率达此值视为"足够优"，参照硬包材早返回阈值

    # AI 调用追踪（供 run_packing_agent 统计 ai_used / tokens）
    _soft_ai_meta = {"used": False, "in_tokens": 0, "out_tokens": 0}

    total_weight_kg  = sum(i.get("weight", 0) for i in items)
    sale_price_usd   = max((i.get("sale_price", 0) for i in items), default=0)
    product_category = next(
        (i.get("product_category", "常规类产品") for i in items if i.get("product_category")),
        "常规类产品",
    )

    # ── 方案二：AI 决定软包材尺寸参数（降级返回 config 默认值）────────────────
    _ai_hf, _ai_sm = _ai_decide_soft_params(
        items, classify_result=classify_result,
        ai_provider=ai_provider, ai_model=ai_model,
        _meta=_soft_ai_meta,
    )

    # ── Step 1: 目录软包材（充当"包材库最优"角色，取 top3）──────────────────
    soft_cands = _prefilter_soft_catalog_bins(
        items, min_protection_rank=min_protection_rank,
        height_factor=_ai_hf, side_margin=_ai_sm,
    )
    catalog_bin = catalog_result = catalog_fee = None
    catalog_util = 0.0
    catalog_tier_rank = len(TIER_ORDER)
    soft_top3_entries = []   # top3 目录袋，格式与硬包材 top3_entries 一致

    if soft_cands:
        def _fh(item):
            return sorted([item["length"], item["width"], item["height"]])[0]
        _stack_h = sum(_fh(i) for i in items)
        # 取前3个候选（已按购买价格升序排列）
        selected_cat = _select_soft_bin_ai(items, soft_cands, ai_provider=ai_provider, ai_model=ai_model)
        _sel_sku     = selected_cat.get("sku")
        _others      = [c for c in soft_cands if c.get("sku") != _sel_sku][:2]
        _top3_cats   = [selected_cat] + _others

        for cat in _top3_cats:
            _bin    = _catalog_to_soft_bin(cat, _stack_h)
            _result = _make_soft_catalog_result(items, _bin)
            _util   = _result["summary"]["avg_utilization"]
            try:
                _fee = calc_bin_fee(_bin, total_weight_kg, sale_price_usd, product_category)
            except Exception:
                _fee = None
            soft_top3_entries.append({
                "bin_data":   _bin,
                "bin_result": {
                    "bin_type":        _bin["type"],
                    "utilization":     _util,
                    "all_placed":      True,
                    "total_bins_used": 1,
                    "unplaced":        [],
                },
                "full": _result,
                "fee":  _fee,
            })

        # ── 关键：从 top3 中按 FBA 最优（费档↑→ 总费用↑）选主推袋，而非按购买价格 ──
        # 这里修复了旧版按购买价排序导致"主推袋 FBA 费用最高"的问题。
        def _fba_sort_key(e):
            f = e["fee"]
            if not f:
                return (999, 9999.0)
            return (tier_rank(f.get("tier", "")), float(f.get("total_fee", 9999)))

        best_cat_entry = min(soft_top3_entries, key=_fba_sort_key)
        catalog_bin    = best_cat_entry["bin_data"]
        catalog_result = best_cat_entry["full"]
        catalog_util   = best_cat_entry["bin_result"]["utilization"]
        catalog_fee    = best_cat_entry["fee"]
        if catalog_fee:
            try:
                catalog_tier_rank = tier_rank(catalog_fee["tier"])
            except Exception:
                pass

    # ── Step 2: 公式定制袋（充当"推荐新包材"角色，使用 AI 决定的参数）────────
    custom_soft = calc_soft_packing(items, height_factor=_ai_hf, side_margin=_ai_sm)
    custom_bin = custom_fee = None
    custom_util = 0.0
    custom_tier_rank = len(TIER_ORDER)

    if custom_soft:
        _cb = custom_soft["bin_data"]
        custom_bin  = {**_cb, "type": "推荐定制软包材"}  # 区分于目录袋
        custom_util = custom_soft["summary"]["avg_utilization"]
        try:
            custom_fee        = calc_bin_fee(custom_bin, total_weight_kg, sale_price_usd, product_category)
            custom_tier_rank  = tier_rank(custom_fee["tier"])
        except Exception:
            pass

    print(f"[soft-recommend] best_catalog={catalog_bin['type'] if catalog_bin else 'None'} "
          f"util={catalog_util:.1%} tier={catalog_fee['tier'] if catalog_fee else 'N/A'} "
          f"fee={catalog_fee['total_fee'] if catalog_fee else 'N/A'}", flush=True)
    print(f"[soft-recommend] custom util={custom_util:.1%} tier={custom_fee['tier'] if custom_fee else 'N/A'} "
          f"fee={custom_fee['total_fee'] if custom_fee else 'N/A'}", flush=True)

    # ── Step 3: winner 决策 ──────────────────────────────────────────────
    # 规则：
    #   a) 目录袋利用率≥80% 且 FBA 费档不劣于定制袋 → 直接用目录袋，无需定制
    #   b) 其余情况：始终用 AI/费用排序对比两者，由 FBA 经济性决定胜负
    #      （移除旧版 catalog_too_loose 利用率阈值——利用率低不代表 FBA 费用高，
    #       不能绕过费用比较直接推定制袋）

    catalog_good = (
        catalog_bin is not None
        and catalog_util >= _SOFT_GOOD_UTIL
        and catalog_tier_rank <= custom_tier_rank
    )

    if catalog_good:
        winner = catalog_bin["type"]
        out_rec_bin = out_rec_fee = None
        out_rec_result_entry = None
        full_rec_result = None
    else:
        # ── 方案一：AI 裁决 winner（降级时按(费档,总费用)排序）────────────────
        _ai_cands = []
        if catalog_bin:
            _ai_cands.append({
                "name":        catalog_bin["type"],
                "tier":        (catalog_fee or {}).get("tier", ""),
                "total_fee":   (catalog_fee or {}).get("total_fee", 999),
                "utilization": catalog_util,
                "cost_price":  catalog_bin.get("cost_price", 0),
            })
        if custom_bin:
            _ai_cands.append({
                "name":        "推荐新包材",
                "tier":        (custom_fee or {}).get("tier", ""),
                "total_fee":   (custom_fee or {}).get("total_fee", 999),
                "utilization": custom_util,
                "cost_price":  0,
            })
        if _ai_cands:
            winner = _ai_select_winner(
                items, _ai_cands, classify_result=classify_result,
                ai_provider=ai_provider, ai_model=ai_model,
                _meta=_soft_ai_meta,
            )
        elif catalog_bin:
            winner = catalog_bin["type"]
        else:
            winner = "推荐新包材"

        out_rec_bin   = custom_bin
        out_rec_fee   = custom_fee
        out_rec_result_entry = {
            "utilization": custom_util, "all_placed": True,
            "total_bins_used": 1, "unplaced": [],
        } if custom_bin else None
        full_rec_result = {**custom_soft, "packed_bins": [
            {**custom_soft["packed_bins"][0], "bin_type": custom_bin["type"]}
        ]} if (custom_soft and custom_bin) else None

        # winner 是目录袋时：清空定制袋输出，不向用户展示劣势方案
        if winner != "推荐新包材":
            out_rec_bin          = None
            out_rec_fee          = None
            out_rec_result_entry = None
            full_rec_result      = None

    # ── 构建 compare_summary ─────────────────────────────────────────────
    # 目录袋充当 best_existing；定制袋充当 recommended（仅不够优时出现）
    compare_summary = {
        "recommended_bin":    out_rec_bin,
        "recommended_result": out_rec_result_entry,
        "best_existing_bin":     catalog_bin["type"] if catalog_bin else "目录无合适软包材",
        "best_existing_result":  {
            "utilization":    catalog_util,
            "all_placed":     catalog_bin is not None,
            "total_bins_used": 1 if catalog_bin else 0,
            "unplaced":       [],
        },
        "winner":                   winner,
        "existing_fee":             catalog_fee,
        "recommended_fee":          out_rec_fee,
        "tier_upgrade":             (custom_fee is not None and catalog_fee is not None
                                     and custom_tier_rank < catalog_tier_rank),
        "soft_bin":                 None,
        "soft_result":              None,
        "soft_fee":                 None,
        "is_soft_mode":             True,
        "best_existing_cost_price": catalog_bin.get("cost_price") if catalog_bin else None,
        "best_existing_sku":        catalog_bin.get("sku")        if catalog_bin else None,
        "top3_existing": [
            {
                "bin_type":   e["bin_data"]["type"],
                "bin":        e["bin_data"],
                "result":     e["bin_result"],
                "fee":        e["fee"],
                "cost_price": e["bin_data"].get("cost_price"),
                "sku":        e["bin_data"].get("sku"),
            }
            for e in soft_top3_entries
        ],
    }

    full_compare = {
        "recommended_bin":    out_rec_bin,
        "recommended_result": full_rec_result,
        "best_existing_bin":  catalog_bin["type"] if catalog_bin else "目录无合适软包材",
        "best_full_result":   catalog_result,
        "top3_full_results":  [e["full"] for e in soft_top3_entries],
        "soft_full_result":   None,
        "compare_summary":    compare_summary,
        "_soft_ai_meta":      _soft_ai_meta,   # 供 run_packing_agent 读取 ai_used
    }

    if winner == "推荐新包材" and full_rec_result:
        primary_result = full_rec_result
    elif catalog_result:
        primary_result = catalog_result
    else:
        primary_result = full_rec_result

    tier_note = (
        f"，可降FBA费档：{catalog_fee.get('tier')}→{custom_fee.get('tier')}"
        if (out_rec_bin and catalog_fee and custom_fee and custom_tier_rank < catalog_tier_rank)
        else ""
    )
    if winner == "推荐新包材" and out_rec_bin:
        w_desc = (f"推荐定制软包材 {out_rec_bin['length']}×{out_rec_bin['width']}×{out_rec_bin['height']}cm"
                  f"（目录袋FBA费偏高，定制袋费档/总费更优）")
    elif catalog_bin:
        w_desc = (f"推荐目录软包材「{catalog_bin['type']}」"
                  f"（{catalog_bin['length']}×{catalog_bin['width']}cm，"
                  f"FBA费{catalog_fee['total_fee'] if catalog_fee else '未知'}元，"
                  f"利用率{catalog_util:.0%}）")
    else:
        w_desc = "目录和公式均无合适软包材"

    agent_str = (
        f"【软包材算法决策】{w_desc}{tier_note}\n"
        f"请严格按照 winner 字段输出总结。\n\n"
        f"详细数据：{json.dumps(compare_summary, ensure_ascii=False)}"
    )
    return agent_str, full_compare, primary_result


def _do_recommend_and_compare(items: list, bins: list = None,
                               ai_provider: str = "anthropic", ai_model: str = None,
                               min_protection_rank: int = 1,
                               classify_result: dict = None) -> tuple:
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
    # 软包材专用路径：所有货物都可软包装时，走独立流程
    # 注意：软包材（袋子）全部 protection_rank=1，min_protection_rank 强制为 1，
    # 避免产品分类 medium/high 防护等级把全部袋子过滤掉。
    if len(items) > 0 and all(i.get("soft_packaging_ok", False) for i in items):
        return _do_recommend_soft(items, ai_provider=ai_provider, ai_model=ai_model,
                                   min_protection_rank=1, classify_result=classify_result)

    available = bins if bins else (
        _prefilter_catalog_bins(items, min_protection_rank=min_protection_rank)
        if _RAW_CATALOG else AVAILABLE_BINS
    )
    max_copies = len(items)

    # 若传入的 bins 全部是 AVAILABLE_BINS 占位箱型（preferred_type 在目录中无匹配时的兜底），
    # 直接扩回全量真实包材，避免占位箱型干扰推荐结果
    _AVAILABLE_BIN_TYPES = {b["type"] for b in AVAILABLE_BINS}
    if bins and _RAW_CATALOG and all(b["type"] in _AVAILABLE_BIN_TYPES for b in bins):
        fallback = _prefilter_catalog_bins(items, min_protection_rank=min_protection_rank)
        if len(fallback) > len(bins):
            logger.info("[recommend] 检测到占位箱型，扩回全量真实包材 %d 个", len(fallback))
            available = fallback

    print(f"[recommend] 扫描包材数: {len(available)}, 前3: {[b['type'] for b in available[:3]]}", flush=True)

    # 提取货物信息：总重(kg)、售价(USD)、产品类别
    total_weight_kg  = sum(i.get("weight", 0) for i in items)
    sale_price_usd   = max((i.get("sale_price", 0) for i in items), default=0)
    product_category = next((i.get("product_category", "常规类产品") for i in items
                             if i.get("product_category")), "常规类产品")

    # ── 体积预筛：贪心扫描前用纯数学快速过滤，将候选箱型从150缩到~25 ─────────────
    # 贪心扫描是 O(n²×箱型数)，箱型数越少越快；纯体积排名 O(箱型数)，毫秒级。
    _SCAN_LIMIT = 25
    if len(available) > _SCAN_LIMIT:
        _total_vol = sum(i["length"] * i["width"] * i["height"] for i in items)
        # 最大单件三边从小到大排序，用于判断箱子能否放下最大件
        _max_dims_sorted = sorted([
            max(i["length"] for i in items),
            max(i["width"]  for i in items),
            max(i["height"] for i in items),
        ])
        _vol_pre = []
        for b in available:
            bvol = b["length"] * b["width"] * b["height"]
            if bvol <= 0:
                continue
            # 任意旋转后最大件能否装入：双方三边各自排序后逐一比较
            b_sorted = sorted([b["length"], b["width"], b["height"]])
            if any(d > bd for d, bd in zip(_max_dims_sorted, b_sorted)):
                continue  # 连最大单件都放不下，直接跳过
            bwt   = b.get("max_weight", MAX_BIN_WEIGHT)
            n_vol = math.ceil(_total_vol / (bvol * 0.65))
            n_wt  = math.ceil(total_weight_kg / bwt) if bwt > 0 else 9999
            n_est = max(n_vol, n_wt, 1)
            util  = _total_vol / (bvol * n_est)
            _vol_pre.append((-n_est * 1000 + util * 100, b))
        _vol_pre.sort(key=lambda x: x[0], reverse=True)
        _before = len(available)
        available = [b for _, b in _vol_pre[:_SCAN_LIMIT]]
        logger.info("[recommend] 体积预筛: %d → %d 个候选箱型", _before, len(available))

    # ── 第一步：扫描现有箱型，找出前3优（贪心快速评分，不运行 OR-Tools）────────────
    # scan_mode=True：跳过 OR-Tools，仅用贪心算法给每个箱型打分（< 100ms/箱型）。
    # 货物超 50 件时用体积分层抽样（50件），让贪心打分保持 O(50²) 而非 O(n²)。
    # 抽样：按体积排序后均匀抽取，保留大中小件的代表性分布。
    _SCAN_SAMPLE = 50
    if len(items) > _SCAN_SAMPLE:
        _sorted_by_vol = sorted(items, key=lambda i: i["length"] * i["width"] * i["height"], reverse=True)
        _step   = len(_sorted_by_vol) / _SCAN_SAMPLE
        _scan_items = [_sorted_by_vol[int(k * _step)] for k in range(_SCAN_SAMPLE)]
    else:
        _scan_items = items

    _scored_bins = []
    for bin_data in available:
        result     = calculate_packing(_scan_items, [bin_data] * len(_scan_items), scan_mode=True)
        all_placed = result["summary"]["all_placed"]
        util       = result["summary"]["avg_utilization"]
        num_bins   = result["summary"]["total_bins_used"]

        if all_placed:
            score = 10000 - num_bins * 10 + util
        else:
            placed_count = sum(b["item_count"] for b in result["packed_bins"])
            score = placed_count * 10 + util

        _scored_bins.append((score, bin_data, {
            "bin_type":        bin_data["type"],
            "utilization":     util,
            "all_placed":      all_placed,
            "total_bins_used": num_bins,
            "unplaced":        result["unplaced_items"],
        }))

    _scored_bins.sort(key=lambda x: x[0], reverse=True)
    _top3 = _scored_bins[:3]

    # 对前3各跑完整 OR-Tools，得到精确坐标；同时计算各自的运费档级
    _scan_total_weight = sum(i.get("weight", 0) for i in items)
    _scan_sale_price   = max((i.get("sale_price", 0) for i in items), default=0)
    _scan_category     = next((i.get("product_category", "常规类产品") for i in items
                               if i.get("product_category")), "常规类产品")
    top3_entries = []  # list of {"bin_data", "bin_result", "full", "fee"}
    for _, bd, br in _top3:
        _full = calculate_packing(items, [bd] * max_copies)
        try:
            _fee = calc_bin_fee(bd, _scan_total_weight, _scan_sale_price, _scan_category)
        except Exception:
            _fee = None
        top3_entries.append({"bin_data": bd, "bin_result": br, "full": _full, "fee": _fee})

    best_bin_result = _top3[0][2] if _top3 else None
    best_bin_data   = _top3[0][1] if _top3 else None
    best_full       = top3_entries[0]["full"] if top3_entries else None

    best_util  = best_bin_result["utilization"]     if best_bin_result else 0
    best_all   = best_bin_result["all_placed"]      if best_bin_result else False
    best_type  = best_bin_result["bin_type"]        if best_bin_result else "无"
    print(f"[recommend] 扫描结果: best={best_type}, util={best_util:.2%}, all_placed={best_all}, top3数={len(top3_entries)}", flush=True)
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
    # 方案二：AI 决定尺寸余量（失败时降级用 config.BIN_SIZE_BUFFER）
    _ai_buffer = _ai_decide_bin_buffer(
        items, classify_result=classify_result,
        ai_provider=ai_provider, ai_model=ai_model,
    )
    rec_bin    = _calc_recommended_bin(items, size_buffer=_ai_buffer)
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
            # 收紧时沿用 AI 决定的 buffer，保持与初始 rec_bin 计算一致
            new_l = min(math.ceil((ax + _ai_buffer) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO,
                        rec_bin["length"])
            new_w = min(math.ceil((ay + _ai_buffer) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO,
                        rec_bin["width"])
            new_h = min(math.ceil((az + _ai_buffer) / BIN_SIZE_ROUND_TO) * BIN_SIZE_ROUND_TO,
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

    # ── 第2.6步：软包材方案（硬路径中不再触发，软路径已在函数顶部分流至 _do_recommend_soft）──
    soft_ok = False
    soft_result = None
    soft_bin_data = None
    soft_fee = None
    soft_tier_rank = len(TIER_ORDER)
    soft_tier_upgrade = False

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
            "top3_existing": [
                {
                    "bin_type":   e["bin_data"]["type"],
                    "bin":        e["bin_data"],
                    "result":     e["bin_result"],
                    "fee":        e["fee"],
                    "cost_price": e["bin_data"].get("cost_price"),
                    "sku":        e["bin_data"].get("sku"),
                }
                for e in top3_entries
            ],
        }
        full_compare = {
            "recommended_bin":    None,
            "recommended_result": None,
            "best_existing_bin":  best_type,
            "best_full_result":   best_full,
            "top3_full_results":  [e["full"] for e in top3_entries],
            "soft_full_result":   soft_result if soft_ok else None,
            "compare_summary":    compare_summary,
        }
        print(f"[recommend] 早返回: winner={best_type!r}, best_full={'None' if best_full is None else 'dict'}", flush=True)
        return json.dumps(compare_summary, ensure_ascii=False), full_compare, best_full

    # ── 第四步：方案一 AI 裁决 winner（降级时按(费档,总费用)排序）─────────────
    _ai_cands = []
    if rec_single_ok:
        _ai_cands.append({
            "name":        "推荐新包材",
            "tier":        (rec_fee      or {}).get("tier",      ""),
            "total_fee":   (rec_fee      or {}).get("total_fee", 999),
            "utilization": rec_util,
            "cost_price":  0,
        })
    if soft_ok:
        _ai_cands.append({
            "name":        "软包材",
            "tier":        (soft_fee     or {}).get("tier",      ""),
            "total_fee":   (soft_fee     or {}).get("total_fee", 999),
            "utilization": (soft_result["summary"]["avg_utilization"] if soft_result else 0),
            "cost_price":  0,
        })
    if best_all:
        _ai_cands.append({
            "name":        best_type,
            "tier":        (existing_fee or {}).get("tier",      ""),
            "total_fee":   (existing_fee or {}).get("total_fee", 999),
            "utilization": best_util,
            "cost_price":  best_bin_data.get("cost_price", 0) if best_bin_data else 0,
        })

    if _ai_cands:
        winner = _ai_select_winner(
            items, _ai_cands, classify_result=classify_result,
            ai_provider=ai_provider, ai_model=ai_model,
        )
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

    # ── 关键：winner 是现有包材时，完全清空定制包材输出 ────────────────────────
    # 定制包材是为"目录里没有合适包材"准备的兜底方案。
    # 若目录包材已经胜出，继续展示一个"输了"的定制方案只会让用户困惑；
    # 直接置空，前端即不渲染该列。
    if winner != "推荐新包材":
        out_rec_bin       = None
        out_rec_fee       = None
        rec_summary_entry = None
        full_rec_result   = None
        tier_upgrade      = False   # winner 是目录包材时降档标记也无意义

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
        "top3_existing": [
            {
                "bin_type":   e["bin_data"]["type"],
                "bin":        e["bin_data"],
                "result":     e["bin_result"],
                "fee":        e["fee"],
                "cost_price": e["bin_data"].get("cost_price"),
                "sku":        e["bin_data"].get("sku"),
            }
            for e in top3_entries
        ],
    }

    full_compare = {
        "recommended_bin":    out_rec_bin,
        "recommended_result": full_rec_result,
        "best_existing_bin":  best_type,
        "best_full_result":   best_full,
        "top3_full_results":  [e["full"] for e in top3_entries],
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

    # 在 JSON 前拼接明确结论，避免 Claude 误读字段名自行推断
    winner_desc = {
        "推荐新包材": f"推荐新包材 {out_rec_bin['length']}×{out_rec_bin['width']}×{out_rec_bin['height']}cm（利用率更高或能降FBA费档）" if out_rec_bin else "推荐新包材",
        "软包材":   f"推荐软包材（利用率或费率优于硬包材）",
    }.get(winner, f"沿用现有包材「{best_type}」（已是最优，无需更换）")
    tier_note = f"，触发FBA费档降档：{existing_fee.get('tier')}→{(out_rec_fee or {}).get('tier')}" if tier_upgrade else ""
    agent_str = (
        f"【算法决策结论】最终推荐：{winner_desc}{tier_note}\n"
        f"注意：recommended_bin 字段是算法计算出的候选新包材尺寸，winner 字段才是最终决策结果。"
        f"请严格按照 winner 字段写总结，不要自行判断哪个更好。\n\n"
        f"详细数据：{json.dumps(compare_summary, ensure_ascii=False)}"
    )
    return agent_str, full_compare, primary_result


# ── 工具执行 ──────────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict, bins: list = None,
                 original_items: list = None,
                 ai_provider: str = "anthropic", ai_model: str = None,
                 min_protection_rank: int = 1,
                 classify_result: dict = None) -> tuple:
    """
    Returns:
        (agent_str, full_result, compare_result)
        agent_str:      返回给 Agent 的精简摘要
        full_result:    完整装箱坐标数据
        compare_result: 对比数据（仅 recommend_and_compare 时有值）
    original_items — 原始货物列表，用于恢复 Claude 工具调用时可能丢失的 soft_packaging_ok 字段
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
        # AI 无需传 items，直接用上下文中的 original_items；若 AI 仍传了则优先用（兼容旧调用）
        _items = tool_input.get("items") or original_items or []
        # 还原 soft_packaging_ok（AI 工具调用时可能丢失该字段）
        if original_items:
            _orig_map = {i["id"]: i.get("soft_packaging_ok", False) for i in original_items}
            for item in _items:
                if _orig_map.get(item.get("id", ""), False):
                    item["soft_packaging_ok"] = True
        # 提取 AI 传入的约束参数，重新筛选候选包材
        tool_preferred  = tool_input.get("preferred_type")
        tool_tight      = tool_input.get("require_tight", False)
        tool_excl       = set(tool_input.get("excluded_skus") or [])
        tool_max_cost   = tool_input.get("max_cost")
        has_constraints = tool_preferred or tool_tight or tool_excl or (tool_max_cost is not None)

        if has_constraints and _RAW_CATALOG:
            effective_bins = _prefilter_catalog_bins(
                _items,
                excluded_skus=tool_excl,
                max_cost=tool_max_cost,
                preferred_type=tool_preferred,
                require_tight=tool_tight,
                min_protection_rank=min_protection_rank,
            )
            logger.info("[execute_tool] Claude约束筛选: preferred=%s tight=%s excl=%d max_cost=%s → %d候选",
                        tool_preferred, tool_tight, len(tool_excl), tool_max_cost, len(effective_bins))
        else:
            effective_bins = bins

        agent_str, full_compare, primary_result = _do_recommend_and_compare(
            _items, bins=effective_bins,
            ai_provider=ai_provider, ai_model=ai_model,
            min_protection_rank=min_protection_rank,
            classify_result=classify_result,
        )
        winner_in_compare = (full_compare or {}).get("compare_summary", {}).get("winner", "?")
        print(f"[execute_tool] primary_result={'None' if primary_result is None else type(primary_result).__name__}, "
              f"full_compare={'None' if full_compare is None else 'dict'}, winner={winner_in_compare}", flush=True)
        return agent_str, primary_result, full_compare

    return json.dumps({"error": f"未知工具: {tool_name}"}), None, None


# ── 带重试的 API 调用（处理 429 限流 / 529 过载）────────────────────────────

def _api_with_retry(max_retries: int = 3, _provider: str = "anthropic", _ds_client=None, **kwargs) -> object:
    """
    通用带重试的 API 调用封装。
    anthropic: 对 429/529 做指数退避重试。
    deepseek:  通过 OpenAI-compatible 接口，对 RateLimitError 做指数退避重试。
    """
    if _provider == "anthropic":
        for attempt in range(max_retries):
            try:
                return client.messages.create(**kwargs, timeout=300)
            except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
                status = getattr(e, "status_code", 0)
                retryable = status in (429, 529)
                if retryable and attempt < max_retries - 1:
                    wait = 2 ** attempt + 1
                    logger.warning("[API重试] status=%d 第%d次，等待%ds: %s",
                                   status, attempt + 1, wait, e)
                    time.sleep(wait)
                else:
                    raise
    else:
        if _provider == "qwen":
            ds = _ds_client or _get_qwen_client()
        else:
            ds = _ds_client or _get_ds_client()
        system   = kwargs.pop("system", None)
        messages = kwargs.pop("messages", [])
        tools    = kwargs.pop("tools", None)
        tc       = kwargs.pop("tool_choice", None)
        _default_model = "qwen3.6-plus" if _provider == "qwen" else "deepseek-v4-flash"
        model    = kwargs.pop("model", _default_model)
        max_tok  = kwargs.pop("max_tokens", 4096)
        oai_messages = _to_oai_messages(messages, system)
        oai_tools    = _to_oai_tools(tools) if tools else None
        oai_tc = "required" if (isinstance(tc, dict) and tc.get("type") == "any") else "auto"
        # 推理模型限制：① 需要更大的 token 预算；② 不支持 tool_choice=required
        _REASONING_MODELS = {"deepseek-v4-pro", "deepseek-v4-flash", "qwen3.6-plus"}
        if model in _REASONING_MODELS:
            max_tok = max(max_tok, 16384)
            if oai_tc == "required":
                oai_tc = "auto"
        call_kw: dict = {"model": model, "messages": oai_messages, "max_tokens": max_tok}
        # Qwen3 默认开启 thinking 模式，对结构化输出任务会产生大量无用推理 token，
        # 导致响应超时。统一关掉，让它像普通模型一样快速返回。
        if _provider == "qwen" and model in _REASONING_MODELS:
            call_kw["extra_body"] = {"enable_thinking": False}
        if oai_tools:
            call_kw["tools"]       = oai_tools
            call_kw["tool_choice"] = oai_tc
        for attempt in range(max_retries):
            try:
                resp = ds.chat.completions.create(**call_kw, timeout=300)
                return _from_oai_resp(resp)
            except _openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt + 1
                    logger.warning("[DeepSeek重试] 第%d次，等待%ds: %s", attempt + 1, wait, e)
                    time.sleep(wait)
                else:
                    raise


def _call_api_simple(messages: list, max_retries: int = 3) -> object:
    """不带 tools 的普通对话调用，用于生成最终说明文字"""
    return _api_with_retry(
        max_retries=max_retries,
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=messages,
    )


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


# ── 产品包装分类 ──────────────────────────────────────────────────────────────

# 关键词规则表（服务端兜底，AI 不可用时使用）
_CLASSIFY_RULES = [
    # (protection_level, suitable_types, 触发关键词列表, reason)
    ("low",    ["袋子"],    ["衣", "裤", "裙", "袜", "内衣", "帽", "手套", "围巾", "毛衣", "外套", "夹克", "服装", "布料", "毛巾", "床单", "被", "枕套"], "服装/纺织类，建议袋子包装"),
    ("high",   ["三层纸箱", "纸箱"], ["玻璃", "陶瓷", "瓷", "杯", "碗", "盘", "壶", "灯泡", "灯管", "镜", "相机", "镜头", "仪器", "手表", "首饰", "珠宝", "钟"], "易碎/高价值类，建议三层纸箱高防护"),
    ("medium", ["三层纸箱", "纸箱"], ["手机", "平板", "电脑", "显示器", "耳机", "键盘", "鼠标", "充电", "适配器", "音响", "路由器", "电子"], "电子类，建议纸箱包装"),
    ("medium", ["纸箱", "三层纸箱"], ["玩具", "模型", "手办", "积木", "拼图"], "玩具/模型类，建议纸箱包装"),
    ("low",    ["袋子", "纸箱"],   ["书", "本", "笔", "文件", "海报", "贴纸", "卡片", "信封", "画册"], "文具/印刷类，轻巧包装即可"),
    ("low",    ["袋子"],   ["食品", "零食", "糖", "茶叶", "咖啡", "调料"], "食品类，防潮袋包装"),
]

def _keyword_classify(product_title: str, product_category: str) -> dict:
    """基于关键词规则的本地分类，AI 不可用时使用。"""
    title = product_title
    if product_category == "服装产品":
        return {"protection_level": "low", "suitable_types": ["袋子"],
                "reason": "服装产品类别，建议袋子包装", "source": "keyword"}
    for level, types, keywords, reason in _CLASSIFY_RULES:
        if any(k in title for k in keywords):
            return {"protection_level": level, "suitable_types": types,
                    "reason": reason, "source": "keyword"}
    return {"protection_level": "medium", "suitable_types": ["纸箱"],
            "reason": "无明显品类特征，默认中等防护", "source": "keyword"}


def classify_for_packaging(product_title: str, product_category: str = "常规类产品",
                            ai_provider: str = "anthropic", ai_model: str = None) -> dict:
    """
    用 AI 根据产品标题+类别判断包装防护级别和适合的包材类型。
    AI 不可用时自动降级为关键词规则兜底。
    Returns 额外字段 source: "ai" | "keyword"
    """
    import re as _re
    if not product_title.strip():
        result = _keyword_classify("", product_category)
        return result

    if ai_provider == "anthropic":
        model          = ai_model or "claude-haiku-4-5-20251001"
        provider_label = "Anthropic"
    elif ai_provider == "qwen":
        model          = ai_model or "qwen3.6-plus"
        provider_label = "通义千问"
    else:
        model          = ai_model or "deepseek-v4-flash"
        provider_label = "DeepSeek"

    prompt = (
        "你是包装工程专家。根据产品信息，判断包装防护级别和适合的包材类型。\n\n"
        f"产品标题：{product_title}\n"
        f"产品FBA类别：{product_category}\n\n"
        "以 JSON 格式回答（只输出 JSON，不要其他内容）：\n"
        '{"protection_level": "high/medium/low", "suitable_types": ["类型1", "类型2"], "reason": "判断依据"}\n\n'
        "suitable_types 只能从以下类型中选择（1-3种）：三层纸箱、纸箱、单层纸箱、袋子\n"
        "protection_level: high=易碎/精密/高价值, medium=一般, low=耐摔/轻巧/低价值"
    )
    try:
        resp = _api_with_retry(
            _provider=ai_provider,
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        m = _re.search(r'\{', text)
        if m:
            result, _ = json.JSONDecoder().raw_decode(text, m.start())
            result["source"]   = "ai"
            result["model"]    = model
            result["provider"] = provider_label
            valid_types = {"三层纸箱", "纸箱", "单层纸箱", "袋子"}
            result["suitable_types"] = [t for t in result.get("suitable_types", []) if t in valid_types] or ["纸箱"]
            result["ai_input_tokens"]  = resp.usage.input_tokens  or 0
            result["ai_output_tokens"] = resp.usage.output_tokens or 0
            return result
    except Exception as e:
        logger.warning("classify_for_packaging AI调用失败，降级关键词兜底: %s", e)

    result = _keyword_classify(product_title, product_category)
    result["ai_input_tokens"]  = 0
    result["ai_output_tokens"] = 0
    result["model"]    = None
    result["provider"] = None
    return result


# ── Agent 主函数 ──────────────────────────────────────────────────────────────

def run_packing_agent(items: list, bins: list = None, constraints: dict = None,
                      extra_bins: list = None, classify_result: dict = None,
                      ai_provider: str = "anthropic", ai_model: str = None) -> dict:
    """
    AI Agent 模式：Claude 调用 recommend_and_compare 工具完成装箱决策。
      1. 发送货物清单给 Claude，仅提供 recommend_and_compare 工具
      2. Claude 根据产品分类结果选择合适约束参数调用工具
      3. 若结果不满足防护需求，Claude 可调整参数重调（最多2次）
      4. Claude 生成中文总结；API 不可用时自动降级为本地逻辑
    classify_result — 产品分类结果（protection_level / suitable_types / reason）
    constraints     — 前端重推约束字典（excluded_skus / max_cost / preferred_type / require_tight）
    """
    c = constraints or {}
    # 从产品分类结果推算最低防护等级
    # high→4（仅纸箱）, medium→3（三层纸箱+纸箱）, low→1（全部）
    _MIN_PROT_RANK = 1
    if classify_result:
        _prot_level = classify_result.get("protection_level", "medium")
        _MIN_PROT_RANK = {"high": 4, "medium": 3, "low": 1}.get(_prot_level, 1)

    # ── 软包材专用短路（置于 _prefilter_catalog_bins 之前，避免扫描 6000+ 硬包材）──
    # 全部货物均可软包材时：跳过 Claude 工具调用循环，直接本地决策（含内部 AI 子调用）
    if len(items) > 0 and all(i.get("soft_packaging_ok", False) for i in items):
        logger.info("[agent] 全软包材路径，直接本地决策（含AI子调用）")
        _packing_results: list = []
        _, _full_compare, _rec_result = _do_recommend_and_compare(
            items, bins=None, ai_provider=ai_provider, ai_model=ai_model,
            min_protection_rank=1, classify_result=classify_result,
        )
        if _rec_result is not None:
            _packing_results.append(_rec_result)
        _agent_summary = _local_summary(items, _rec_result, _full_compare)
        _fr_bin = (
            (_rec_result or {}).get("packed_bins", [{}])[0].get("bin_type", "无")
            if _rec_result else "无"
        )
        # 从 _full_compare 读取软包材路径的 AI 调用情况（由 _do_recommend_soft 写入）
        _smeta   = (_full_compare or {}).get("_soft_ai_meta", {})
        _sai_used    = bool(_smeta.get("used", False))
        _sai_in_tok  = _smeta.get("in_tokens",  None) if _sai_used else None
        _sai_out_tok = _smeta.get("out_tokens", None) if _sai_used else None
        # 优先从 _meta 读取实际调用的模型名；若无则根据 provider 推断
        _smeta_model = _smeta.get("model")
        if _smeta_model:
            _sai_model    = _smeta_model
            _sai_provider = "Anthropic" if ai_provider == "anthropic" else ("通义千问" if ai_provider == "qwen" else "DeepSeek")
        elif ai_provider == "anthropic":
            _sai_model, _sai_provider = "claude-haiku-4-5-20251001", "Anthropic"
        else:
            _sai_model, _sai_provider = "deepseek-chat", "DeepSeek"
        print(f"[agent] 软包材短路: final_bin={_fr_bin}, ai_used={_sai_used}, "
              f"tokens={_sai_in_tok}in/{_sai_out_tok}out", flush=True)
        return {
            "success":          True,
            "agent_summary":    _agent_summary,
            "ai_used":          _sai_used,
            "ai_error":         None,
            "packing_results":  _packing_results,
            "final_result":     _rec_result,
            "compare_result":   _full_compare,
            "ai_model":         _sai_model    if _sai_used else None,
            "ai_provider":      _sai_provider if _sai_used else None,
            "ai_input_tokens":  _sai_in_tok,
            "ai_output_tokens": _sai_out_tok,
        }

    available = bins if bins else (
        _prefilter_catalog_bins(
            items,
            excluded_skus=set(c.get("excluded_skus", [])),
            max_cost=c.get("max_cost"),
            preferred_type=c.get("preferred_type"),
            require_tight=c.get("require_tight", False),
            min_protection_rank=_MIN_PROT_RANK,
        ) if _RAW_CATALOG else AVAILABLE_BINS
    )
    # 追加用户自定义包材（按 type 去重，自定义的排在最前方便 AI 优先看到）
    if extra_bins:
        existing_types = {b["type"] for b in available}
        new_bins = [b for b in extra_bins if b["type"] not in existing_types]
        if new_bins:
            available = new_bins + available
            logger.info("[agent] 追加用户自定义包材 %d 个，当前候选共 %d 个", len(new_bins), len(available))
    print(f"[agent] run_packing_agent: {len(items)}件货物, available={len(available)}个包材, bins_param={'自定义' if bins else '目录扫描'}, extra_bins={len(extra_bins) if extra_bins else 0}个", flush=True)

    # 根据产品分类结果生成策略引导
    # 用户明确选定的包材类型（preferred_type）锁定后 AI 不得覆盖
    _user_preferred = c.get("preferred_type")   # 前端用户选择的类型芯片
    classify_ctx = ""
    if classify_result:
        level  = classify_result.get("protection_level", "medium")
        types  = classify_result.get("suitable_types") or []
        reason = classify_result.get("reason", "")
        # preferred_type 候选值仅从当前有效类型中选：三层纸箱/纸箱/单层纸箱/袋子
        _default_type = types[0] if types else "纸箱"
        level_guide = {
            "high":   f"产品易碎/高价值，首次调用请设 require_tight=true，preferred_type 选'{_default_type}'",
            "medium": "产品防护需求中等，使用默认参数；若利用率低于40%可尝试 require_tight=true",
            "low":    f"产品防护需求低，preferred_type 可设为 {'或'.join(types[:2]) if types else '纸箱'}",
        }.get(level, "")
        # 若用户已明确选定类型，提示 AI 强制使用该类型
        if _user_preferred:
            level_guide = f"用户已指定包材类型「{_user_preferred}」，preferred_type 必须设为'{_user_preferred}'，不得更改。"
        classify_ctx = (
            f"\n【产品分析】防护级别：{level} | 建议类型：{'/'.join(types)} | 依据：{reason}\n"
            f"【首次调用策略】{level_guide}\n"
            "【迭代说明】若首次结果不符合防护需求或利用率低于40%，"
            "可调整 require_tight/excluded_skus 参数再调用一次工具（preferred_type 不变）。\n"
        )

    system_prompt = (
        "你是三维装箱专家AI助手。你只有一个工具：recommend_and_compare。\n"
        "调用工具时可传入约束参数优化结果：\n"
        "- preferred_type：限定包材类型，可选值：三层纸箱/纸箱/单层纸箱/袋子\n"
        "- require_tight：true=要求箱子贴合货物，适合高防护产品\n"
        "- excluded_skus：排除不满意的包材 SKU\n"
        "- max_cost：包材成本上限（元）\n"
        + classify_ctx +
        "工具返回后评估结果：满足需求则输出简洁中文总结（推荐包材、利用率、FBA费档级）；"
        "不满足则调整参数重调一次。总调用上限2次。"
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
    ai_used               = False   # 标记本次是否真正经过 AI 决策
    ai_error              = None    # 记录 API 失败原因
    if ai_provider == "anthropic":
        _ai_model    = ai_model or "claude-haiku-4-5-20251001"
        _ai_provider = "Anthropic"
    elif ai_provider == "qwen":
        _ai_model    = ai_model or "qwen3.6-plus"
        _ai_provider = "通义千问"
    else:
        _ai_model    = ai_model or "deepseek-v4-flash"
        _ai_provider = "DeepSeek"
    _total_input_tokens   = 0
    _total_output_tokens  = 0

    try:
        for turn in range(10):          # 最多 10 轮，防止无限循环
            # 还没拿到计算结果前强制调用工具，防止 Claude 只输出文字就结束
            tool_choice = {"type": "any"} if final_result is None else {"type": "auto"}
            logger.info("[Agent 第%d轮] tool_choice=%s", turn + 1, tool_choice["type"])

            response = _api_with_retry(
                _provider   = ai_provider,
                model       = _ai_model,
                max_tokens  = 4096,
                system      = system_prompt,
                tools       = TOOLS,
                tool_choice = tool_choice,
                messages    = messages,
            )

            # 累计 token 用量
            if response.usage:
                _total_input_tokens  += response.usage.input_tokens  or 0
                _total_output_tokens += response.usage.output_tokens or 0

            # ── 将 assistant 回复追加到历史 ──
            asst_msg = {"role": "assistant", "content": response.content}
            if hasattr(response, "_raw_oai_msg"):
                asst_msg["_raw_oai_msg"] = response._raw_oai_msg
            messages.append(asst_msg)

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
                    result_str, full, cr = execute_tool(tu.name, tu.input, bins=available,
                                                           original_items=items,
                                                           ai_provider=ai_provider, ai_model=ai_model,
                                                           min_protection_rank=_MIN_PROT_RANK,
                                                           classify_result=classify_result)
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

    # ── 兜底：若 AI 没有执行计算（API 不可用），本地直接跑推荐逻辑 ──────────────
    if final_result is None or compare_result is None:
        _, full_compare, rec_result = _do_recommend_and_compare(items, bins=bins,
                                                                  ai_provider=ai_provider, ai_model=ai_model,
                                                                  min_protection_rank=_MIN_PROT_RANK,
                                                                  classify_result=classify_result)
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
    print(f"[agent] 最终结果: final_bin={fr_bin}, ai_used={ai_used}, ai_error={ai_error}, "
          f"tokens={_total_input_tokens}in/{_total_output_tokens}out", flush=True)
    return {
        "success":           True,
        "agent_summary":     agent_summary,
        "ai_used":           ai_used,
        "ai_error":          ai_error,
        "packing_results":   packing_results,
        "final_result":      final_result,
        "compare_result":    compare_result,
        "ai_model":          _ai_model    if ai_used else None,
        "ai_provider":       _ai_provider if ai_used else None,
        "ai_input_tokens":   _total_input_tokens  if ai_used else None,
        "ai_output_tokens":  _total_output_tokens if ai_used else None,
    }
