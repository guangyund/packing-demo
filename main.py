"""
三维装箱服务 - FastAPI Web 服务
接口：
  POST /api/pack          直接装箱（手动指定箱型）
  POST /api/agent-pack    AI Agent 自动选箱并计算
  GET  /api/results/{id}  查询装箱结果
  GET  /view              Three.js 3D 可视化页面
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 加载 .env（VOYAGE_API_KEY、ANTHROPIC_API_KEY 等）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

import logging
import uuid
import re
import json
import base64
import asyncio
import datetime
import httpx
import pymysql
import pymysql.cursors
import anthropic
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from packing_engine import calculate_packing, AVAILABLE_BINS
from agent import run_packing_agent, _calc_recommended_bin
from monitor import log_anomaly

# ── 包材目录（已迁移至 MySQL bins_catalog 表，通过 _get_db() 查询）─────────────
from material_advisor import (
    get_fragility, recommend_materials,
    calc_packed_dimensions, calc_material_cost,
)
from vector_store import search_materials, count_materials
from embedder import get_provider, set_provider

logger = logging.getLogger(__name__)

app = FastAPI(title="3D Bin Packing Service", version="1.0.0")

# 静态文件目录（viewer.html）
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 内存结果存储（热缓存，重启后从文件恢复）
_results: dict = {}

# Session 级别计时：记录每个 session 第一次请求的开始时间
# key: session_id, value: {"start_ts": float, "call_count": int}
_session_timing: dict = {}

# 产品分类结果服务端缓存：key=session_id, value=classify_result dict
# 前端只传 session_id，服务端自己取，classify_result 不再经过前端中转
_classify_session_cache: dict = {}

# 装箱结果持久化目录
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

# 反馈存储 - MySQL
_DB_CONFIG = {
    "host":        "127.0.0.1",
    "port":        3306,
    "user":        "root",
    "password":    "Deng123456*",
    "database":    "packing_demo",
    "charset":     "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

def _get_db():
    return pymysql.connect(**_DB_CONFIG)


# ── 数据模型 ──────────────────────────────────────────────────────────────────

class ItemModel(BaseModel):
    id: str
    length: float
    width: float
    height: float
    weight: float
    product_title: str = ""              # 产品标题/品名，供AI判断包装防护级别和适合类型
    sale_price: float = 0.0             # 产品售价（USD），用于运费档级计算
    product_category: str = "常规类产品"  # 常规类产品 / 服装产品 / 危险品（FBA费档用）
    soft_packaging_ok: bool = False      # 是否允许使用软包材（如多层袋）


class BinModel(BaseModel):
    type: str
    length: float
    width: float
    height: float
    max_weight: float


class PackRequest(BaseModel):
    items: List[ItemModel]
    bins: List[BinModel]


class AgentPackRequest(BaseModel):
    items: List[ItemModel]
    bins: Optional[List[BinModel]] = None
    user_bins: Optional[List[BinModel]] = None   # 仅用于存档，不影响 AI 计算
    excluded_skus:   Optional[List[str]] = None  # 重推时排除的 SKU
    max_cost:        Optional[float] = None       # 成本上限（元）
    preferred_type:  Optional[str]  = None        # 偏好包材类型关键词
    require_tight:   bool = False                 # 是否要求紧凑匹配（太大原因）
    classify_result:         Optional[dict] = None  # 产品分类结果，用于引导 AI 策略
    session_id:              Optional[str]  = None  # 计算会话ID（前端生成，同一次计算多个方案共享）
    plan_type:               Optional[str]  = None  # 本方案的包材类型标签（如"纸箱"/"袋子"）
    classify_input_tokens:   Optional[int]  = None  # 防护分析 AI 输入 token（仅第一个方案传入）
    classify_output_tokens:  Optional[int]  = None  # 防护分析 AI 输出 token（仅第一个方案传入）
    classify_source:         Optional[str]  = None  # ai=AI调用 keyword=本地规则（仅第一个方案传入）
    classify_model:          Optional[str]  = None  # 防护分析使用的AI模型（仅第一个方案传入）
    classify_provider:       Optional[str]  = None  # 防护分析 AI 厂商（仅第一个方案传入）
    ai_provider:             Optional[str]  = None  # 用户选择的 AI 厂商（anthropic/deepseek）
    ai_model:                Optional[str]  = None  # 用户选择的具体模型（可选，留空则使用默认）


class ImageView(BaseModel):
    image_base64: str
    media_type: str = "image/jpeg"
    view: str = "正面"


class RecognizeRequest(BaseModel):
    images: List[ImageView]


class Generate3DRequest(BaseModel):
    image_base64: str
    media_type: str = "image/jpeg"


class Tripo3DRequest(BaseModel):
    image_base64: str
    media_type: str = "image/jpeg"
    # api_key 已移至服务端 .env（TRIPO3D_API_KEY），不再由前端传入


class FeedbackModel(BaseModel):
    result_id: str
    session_id: Optional[str] = None
    calc_no: Optional[str] = None
    plan_no: Optional[str] = None
    plan_type: Optional[str] = None
    recommended_bin: Optional[str] = None
    recommended_sku: Optional[str] = None
    selected_plan: Optional[str] = None       # 选定的方案：rec / soft / best
    selected_rank: Optional[int] = None       # 现有包材排名 1/2/3，仅 selected_plan=best 时有效
    selection_method: Optional[str] = None    # default / auto / manual
    adopted: Optional[bool] = None            # 实际是否采用（反馈时填写，选择事件可为 null）
    actual_used_bin: Optional[str] = None
    actual_used_sku: Optional[str] = None
    reason_changed: Optional[str] = None
    reason_detail: Optional[str] = None
    items_summary: Optional[List[dict]] = None
    operator_id: Optional[str] = None


class ClassifyRequest(BaseModel):
    product_title: str = ""
    product_category: str = "常规类产品"
    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None
    session_id: Optional[str] = None   # 传入后结果缓存于服务端，agent-pack 凭此取用


class FeedbackUpdateModel(BaseModel):
    """PATCH 时只传需要更新的字段"""
    selected_plan:    Optional[str]  = None
    selected_rank:    Optional[int]  = None
    selection_method: Optional[str]  = None
    recommended_bin:  Optional[str]  = None
    recommended_sku:  Optional[str]  = None
    adopted:          Optional[bool] = None
    actual_used_bin:  Optional[str]  = None
    reason_changed:   Optional[str]  = None
    reason_detail:    Optional[str]  = None


class OptimizationFeedbackModel(BaseModel):
    result_id:   Optional[str] = None
    category:    str           = "其他"
    content:     str
    operator_id: Optional[str] = None


class SmartPackRequest(BaseModel):
    product_category: str
    fragility_level: str           # high / medium / low
    length: float                  # 产品净尺寸 cm
    width: float
    height: float
    weight: float                  # kg
    quantity: int = 1


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _generate_calc_no() -> str:
    """生成计算编号：BZTJ + YY + MMDD + HHmmss，北京时间"""
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    return now.strftime("BZTJ%y%m%d%H%M%S")

def _save_result(result: dict) -> str:
    """保存装箱结果到内存和文件，返回唯一 ID"""
    result_id = str(uuid.uuid4())
    _results[result_id] = result
    file_path = os.path.join(_RESULTS_DIR, f"{result_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return result_id


def _save_pack_result(result_id: str, agent_result: dict, input_bins: list | None = None,
                      session_id: str | None = None, plan_type: str | None = None,
                      classify_result: dict | None = None,
                      duration_ms: int | None = None,
                      classify_input_tokens: int | None = None,
                      classify_output_tokens: int | None = None,
                      classify_source: str | None = None,
                      classify_model: str | None = None,
                      classify_provider: str | None = None) -> tuple[str, str]:
    """将装箱关键指标写入 pack_results 表，返回 (calc_no, plan_no)"""
    try:
        compare = agent_result.get("compare_result") or {}
        summary = compare.get("compare_summary") or {}
        final   = agent_result.get("final_result") or {}

        winner      = summary.get("winner", "")
        winner_fee  = summary.get("recommended_fee") or summary.get("soft_fee") or summary.get("existing_fee") or {}
        exist_fee   = summary.get("existing_fee") or {}

        winner_bin  = ""
        winner_sku  = ""
        if winner == "推荐新包材":
            rb = summary.get("recommended_bin") or {}
            winner_bin = rb.get("type", "推荐新包材")
            winner_sku = ""  # 推荐新包材是定制尺寸，无SKU
            winner_fee = summary.get("recommended_fee") or {}
        elif winner == "软包材":
            sb = summary.get("soft_bin") or {}
            winner_bin = sb.get("type") or f"软包材 {sb.get('length','')}×{sb.get('width','')}×{sb.get('height','')}cm"
            winner_sku = sb.get("sku") or ""
            winner_fee = summary.get("soft_fee") or {}
        else:
            winner_bin = summary.get("best_existing_bin", "")
            winner_sku = summary.get("best_existing_sku") or ""
            winner_fee = exist_fee

        winner_tier      = winner_fee.get("tier", "")
        winner_total_fee = winner_fee.get("total_fee")
        exist_tier       = exist_fee.get("tier", "")
        exist_total_fee  = exist_fee.get("total_fee")

        tier_upgraded = None
        fee_saved     = None
        from shipping_tiers import tier_rank, TIER_ORDER
        if winner_tier and exist_tier:
            tier_upgraded = 1 if tier_rank(winner_tier) < tier_rank(exist_tier) else 0
        if winner_total_fee is not None and exist_total_fee is not None:
            fee_saved = round(float(exist_total_fee) - float(winner_total_fee), 2)

        packed_bins    = final.get("packed_bins", [])
        utilization    = packed_bins[0].get("utilization") if packed_bins else None
        item_count     = sum(b.get("item_count", 0) for b in packed_bins) if packed_bins else None
        total_weight   = sum(b.get("total_weight", 0) for b in packed_bins) if packed_bins else None
        product_category = next(
            (i.get("product_category") for i in (agent_result.get("items") or []) if i.get("product_category")),
            "常规类产品"
        )

        ai_model          = agent_result.get("ai_model")
        ai_provider       = agent_result.get("ai_provider")
        ai_error          = str(agent_result.get("ai_error"))[:500] if agent_result.get("ai_error") else None
        ai_input_tokens   = agent_result.get("ai_input_tokens")
        ai_output_tokens  = agent_result.get("ai_output_tokens")

        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        conn = _get_db()
        try:
            with conn.cursor() as cur:
                # 生成 calc_no / plan_no
                calc_no = None
                plan_no = None
                if session_id:
                    cur.execute(
                        "SELECT calc_no FROM pack_results WHERE session_id=%s AND calc_no IS NOT NULL LIMIT 1",
                        (session_id,)
                    )
                    row = cur.fetchone()
                    calc_no = row["calc_no"] if row else _generate_calc_no()
                    cur.execute(
                        "SELECT COUNT(*) AS cnt FROM pack_results WHERE session_id=%s AND plan_no IS NOT NULL",
                        (session_id,)
                    )
                    cnt = (cur.fetchone() or {}).get("cnt", 0)
                    plan_no = calc_no + chr(65 + cnt)
                else:
                    calc_no = _generate_calc_no()
                    plan_no = calc_no + "A"

                top3_existing_json = json.dumps(
                    summary.get("top3_existing") or [], ensure_ascii=False
                )
                cur.execute("""
                    INSERT INTO pack_results
                      (result_id, session_id, calc_no, plan_no, plan_type,
                       winner, winner_bin, winner_sku, winner_tier, winner_total_fee,
                       existing_bin, existing_tier, existing_total_fee,
                       tier_upgraded, fee_saved, utilization, item_count, total_weight,
                       product_category, ai_used, ai_model, ai_provider, ai_error,
                       ai_input_tokens, ai_output_tokens,
                       classify_input_tokens, classify_output_tokens,
                       classify_source, classify_model, classify_provider,
                       top3_existing_json, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      session_id=%s, calc_no=%s, plan_no=%s, plan_type=%s,
                      winner=%s, winner_bin=%s, winner_tier=%s, winner_total_fee=%s,
                      existing_tier=%s, existing_total_fee=%s, tier_upgraded=%s,
                      fee_saved=%s, utilization=%s, ai_used=%s,
                      ai_model=%s, ai_provider=%s, ai_error=%s,
                      ai_input_tokens=%s, ai_output_tokens=%s,
                      classify_input_tokens=%s, classify_output_tokens=%s,
                      classify_source=%s, classify_model=%s, classify_provider=%s,
                      top3_existing_json=%s
                """, (
                    result_id, session_id, calc_no, plan_no, plan_type,
                    winner, winner_bin, winner_sku, winner_tier, winner_total_fee,
                    summary.get("best_existing_bin", ""), exist_tier, exist_total_fee,
                    tier_upgraded, fee_saved, utilization, item_count, total_weight,
                    product_category, 1 if agent_result.get("ai_used") else 0,
                    ai_model, ai_provider, ai_error, ai_input_tokens, ai_output_tokens,
                    classify_input_tokens, classify_output_tokens,
                    classify_source, classify_model, classify_provider,
                    top3_existing_json, now,
                    session_id, calc_no, plan_no, plan_type,
                    winner, winner_bin, winner_tier, winner_total_fee,
                    exist_tier, exist_total_fee, tier_upgraded,
                    fee_saved, utilization, 1 if agent_result.get("ai_used") else 0,
                    ai_model, ai_provider, ai_error, ai_input_tokens, ai_output_tokens,
                    classify_input_tokens, classify_output_tokens,
                    classify_source, classify_model, classify_provider,
                    top3_existing_json,
                ))

                # 货品明细
                items_rows = agent_result.get("items") or []
                if items_rows:
                    cur.executemany("""
                        INSERT INTO pack_result_items
                          (result_id, item_id, length, width, height, weight,
                           product_title, sale_price, product_category, soft_packaging_ok)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, [
                        (result_id,
                         it.get("id", ""),
                         it.get("length", 0), it.get("width", 0), it.get("height", 0),
                         it.get("weight", 0),
                         it.get("product_title", ""),
                         it.get("sale_price", 0),
                         it.get("product_category", ""),
                         1 if it.get("soft_packaging_ok") else 0)
                        for it in items_rows
                    ])

                # 用户自填包材
                if input_bins:
                    cur.executemany("""
                        INSERT INTO pack_result_input_bins
                          (result_id, type, length, width, height, max_weight)
                        VALUES (%s,%s,%s,%s,%s,%s)
                    """, [
                        (result_id,
                         b.get("type", ""),
                         b.get("length", 0), b.get("width", 0), b.get("height", 0),
                         b.get("max_weight", 0))
                        for b in input_bins
                    ])

                # 方案明细存档
                cur.execute("""
                    INSERT INTO pack_scheme_detail
                      (calc_no, plan_no, session_id, plan_type, classify_result, agent_summary,
                       final_result, compare_result, created_at, duration_ms)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      plan_type=%s, classify_result=%s,
                      agent_summary=%s, final_result=%s, compare_result=%s, duration_ms=%s
                """, (
                    calc_no, plan_no, session_id,
                    plan_type,
                    json.dumps(classify_result, ensure_ascii=False) if classify_result else None,
                    agent_result.get("agent_summary"),
                    json.dumps(agent_result.get("final_result"),   ensure_ascii=False),
                    json.dumps(agent_result.get("compare_result"), ensure_ascii=False),
                    now, duration_ms,
                    plan_type,
                    json.dumps(classify_result, ensure_ascii=False) if classify_result else None,
                    agent_result.get("agent_summary"),
                    json.dumps(agent_result.get("final_result"),   ensure_ascii=False),
                    json.dumps(agent_result.get("compare_result"), ensure_ascii=False),
                    duration_ms,
                ))
            conn.commit()
        finally:
            conn.close()
        return calc_no, plan_no
    except Exception as e:
        import traceback
        print("【_save_pack_result 异常】", e, flush=True)
        traceback.print_exc()
        logger.warning("写入 pack_results 失败（不影响主流程）: %s", e)
        return _generate_calc_no(), result_id[:8]  # fallback，不影响主流程


def _load_result(result_id: str) -> dict | None:
    """从内存或文件加载装箱结果"""
    if result_id in _results:
        return _results[result_id]
    file_path = os.path.join(_RESULTS_DIR, f"{result_id}.json")
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            result = json.load(f)
        _results[result_id] = result  # 回填内存缓存
        return result
    return None


# ── 接口 ──────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackModel):
    """创建反馈记录（每个 result_id 唯一一条，重复 POST 会覆盖）"""
    rid = feedback.result_id
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    d = feedback.model_dump()
    items_json = json.dumps(d.get("items_summary"), ensure_ascii=False) if d.get("items_summary") else None
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT created_at FROM feedback WHERE result_id=%s", (rid,))
            row = cur.fetchone()
            created_at = row["created_at"] if row else now
            cur.execute("""
                INSERT INTO feedback
                  (result_id,session_id,calc_no,plan_no,plan_type,recommended_bin,recommended_sku,
                   selected_plan,selected_rank,selection_method,
                   adopted,actual_used_bin,actual_used_sku,reason_changed,reason_detail,
                   items_summary,operator_id,created_at,updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                  session_id=VALUES(session_id), calc_no=VALUES(calc_no), plan_no=VALUES(plan_no),
                  plan_type=VALUES(plan_type),
                  recommended_bin=VALUES(recommended_bin), recommended_sku=VALUES(recommended_sku),
                  selected_plan=VALUES(selected_plan), selected_rank=VALUES(selected_rank),
                  selection_method=VALUES(selection_method),
                  adopted=VALUES(adopted), updated_at=VALUES(updated_at)
            """, (rid, d.get("session_id"), d.get("calc_no"), d.get("plan_no"), d.get("plan_type"),
                  d.get("recommended_bin"), d.get("recommended_sku"),
                  d.get("selected_plan"), d.get("selected_rank"), d.get("selection_method"), d.get("adopted"),
                  d.get("actual_used_bin"), d.get("actual_used_sku"),
                  d.get("reason_changed"), d.get("reason_detail"),
                  items_json, d.get("operator_id"), created_at, now))
        conn.commit()
    finally:
        conn.close()
    return {"success": True, "result_id": rid}


@app.patch("/api/feedback/{result_id}")
async def update_feedback(result_id: str, update: FeedbackUpdateModel):
    """更新已有反馈记录的部分字段（静默自动选择 / 手动选择 / 最终确认）"""
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    patch = {k: v for k, v in update.model_dump().items() if v is not None}
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT result_id FROM feedback WHERE result_id=%s", (result_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="反馈记录不存在")
            if patch:
                patch["updated_at"] = now
                set_clause = ", ".join(f"{k}=%s" for k in patch)
                cur.execute(f"UPDATE feedback SET {set_clause} WHERE result_id=%s",
                            list(patch.values()) + [result_id])
        conn.commit()
    finally:
        conn.close()
    return {"success": True, "result_id": result_id}


@app.get("/api/feedback/stats")
async def feedback_stats():
    """返回反馈采纳率统计，按推荐包材分组"""
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as total, SUM(adopted=1) as adopted FROM feedback")
            summary = cur.fetchone()
            total   = summary["total"] or 0
            adopted = int(summary["adopted"] or 0)
            cur.execute("""
                SELECT recommended_bin, COUNT(*) as cnt,
                       SUM(adopted=1) as adopted_cnt, reason_changed
                FROM feedback GROUP BY recommended_bin, reason_changed
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    by_bin: dict = {}
    for r in rows:
        bin_name = r["recommended_bin"] or "未知"
        if bin_name not in by_bin:
            by_bin[bin_name] = {"total": 0, "adopted": 0, "reasons": {}}
        by_bin[bin_name]["total"]   += r["cnt"]
        by_bin[bin_name]["adopted"] += int(r["adopted_cnt"] or 0)
        if r["reason_changed"]:
            rc = r["reason_changed"]
            by_bin[bin_name]["reasons"][rc] = by_bin[bin_name]["reasons"].get(rc, 0) + r["cnt"]

    for b in by_bin.values():
        b["adoption_rate"] = round(b["adopted"] / b["total"], 3) if b["total"] > 0 else 0.0

    return {
        "total":         total,
        "adopted":       adopted,
        "adoption_rate": round(adopted / total, 3) if total > 0 else 0.0,
        "by_bin":        by_bin,
    }


@app.post("/api/optimization-feedback")
async def submit_optimization_feedback(fb: OptimizationFeedbackModel):
    """提交系统优化反馈，存入 optimization_feedback 表"""
    if not fb.content or not fb.content.strip():
        raise HTTPException(status_code=400, detail="content 不能为空")
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO optimization_feedback (result_id, category, content, operator_id, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (fb.result_id, fb.category or "其他", fb.content.strip(), fb.operator_id, now),
            )
        conn.commit()
    finally:
        conn.close()
    return {"success": True}


@app.get("/api/bins-catalog")
async def bins_catalog(
    q:    str = "",
    type: str = "",
    page: int = 1,
    page_size: int = 50,
):
    """
    查询包材目录（从 MySQL bins_catalog 表）。
    q        — 关键词搜索（SKU / 名称 / 包材类型）
    type     — 过滤类型：硬包材 / 软包材 / 空字符串=全部
    page     — 页码（从 1 开始）
    page_size — 每页条数（最多 200）
    """
    page_size = min(page_size, 200)
    offset    = (page - 1) * page_size

    where_clauses = []
    params: list  = []

    if type in ("硬包材", "软包材"):
        where_clauses.append("bin_type = %s")
        params.append(type)
    if q:
        like = f"%{q}%"
        where_clauses.append("(sku LIKE %s OR name LIKE %s OR mat_type LIKE %s)")
        params.extend([like, like, like])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM bins_catalog {where_sql}", params)
            total = cur.fetchone()["cnt"]

            cur.execute(
                f"SELECT sku, name, price, length, width, height, "
                f"       bin_type AS `type`, mat_type AS mat_name, "
                f"       protection_level, protection_rank, max_weight "
                f"FROM bins_catalog {where_sql} "
                f"ORDER BY bin_type, mat_type, length "
                f"LIMIT %s OFFSET %s",
                params + [page_size, offset],
            )
            items = cur.fetchall()
    finally:
        conn.close()

    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "items":     items,
    }


@app.post("/api/pack")
async def pack(request: PackRequest):
    """直接装箱计算（需手动指定箱型）"""
    try:
        items = [item.model_dump() for item in request.items]
        bins  = [b.model_dump() for b in request.bins]
        result = calculate_packing(items, bins)

        result_id = _save_result(result)
        return {
            "success": True,
            **result,
            "result_id": result_id,
            "view_url": f"/view?id={result_id}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classify-product")
async def classify_product(request: ClassifyRequest):
    """用 AI 判断产品包装防护级别和适合的包材类型。
    若传入 session_id，分类结果缓存于服务端；后续 agent-pack 凭 session_id 取用，
    前端无需将 classify_result 回传给服务端。
    """
    from agent import classify_for_packaging
    try:
        result = classify_for_packaging(
            request.product_title, request.product_category,
            ai_provider=request.ai_provider or "anthropic",
            ai_model=request.ai_model,
        )
        # 服务端缓存分类结果，key=session_id
        if request.session_id:
            _classify_session_cache[request.session_id] = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_AGENT_PACK_SINGLE_TIMEOUT_MS  = 30_000   # 单次请求超 30s 记录超时
_AGENT_PACK_SESSION_TIMEOUT_MS = 90_000   # session 累计超 90s 记录超时

@app.post("/api/agent-pack")
async def agent_pack(request: AgentPackRequest):
    """AI Agent 自动选箱并计算装箱方案"""
    import time as _time
    _t0 = _time.time()
    sid = request.session_id

    # ── Session 计时：首次请求记录开始时间 ────────────────────────────────────
    if sid:
        if sid not in _session_timing:
            _session_timing[sid] = {"start_ts": _t0, "call_count": 0}
        _session_timing[sid]["call_count"] += 1

    try:
        items = [item.model_dump() for item in request.items]
        bins      = [b.model_dump() for b in request.bins]      if request.bins      else None
        user_bins = [b.model_dump() for b in request.user_bins] if request.user_bins else None
        constraints = None
        if any([request.excluded_skus, request.max_cost, request.preferred_type, request.require_tight]):
            constraints = {
                "excluded_skus": request.excluded_skus or [],
                "max_cost":      request.max_cost,
                "preferred_type": request.preferred_type,
                "require_tight": request.require_tight,
            }

        # ── classify_result 优先从服务端 session 缓存取，不信任前端回传的值 ──────
        # 若服务端缓存有该 session 的分类结果，直接使用；
        # 否则降级接受前端传入的值（兼容本地规则兜底、旧版前端）
        _srv_classify = _classify_session_cache.get(sid) if sid else None
        effective_classify = _srv_classify or request.classify_result

        # classify 统计字段同样从服务端缓存取，不再由前端传入
        _classify_meta = effective_classify or {}
        _classify_in_tok  = _classify_meta.get("ai_input_tokens")  if _srv_classify else request.classify_input_tokens
        _classify_out_tok = _classify_meta.get("ai_output_tokens") if _srv_classify else request.classify_output_tokens
        _classify_source  = _classify_meta.get("source")           if _srv_classify else request.classify_source
        _classify_model   = _classify_meta.get("model")            if _srv_classify else request.classify_model
        _classify_provider= _classify_meta.get("provider")         if _srv_classify else request.classify_provider

        agent_result = run_packing_agent(items, bins=bins, constraints=constraints,
                                          extra_bins=user_bins, classify_result=effective_classify,
                                          ai_provider=request.ai_provider or "anthropic",
                                          ai_model=request.ai_model)

        _duration_ms = int((_time.time() - _t0) * 1000)

        # mat_type：plan_type 优先，单类型旧计算用 preferred_type 兜底
        mat_type = request.plan_type or request.preferred_type

        calc_no = None
        if agent_result.get("success") and agent_result.get("final_result"):
            result_id = _save_result(agent_result["final_result"])
            agent_result["result_id"] = result_id
            agent_result["view_url"] = f"/view?id={result_id}"
            agent_result["items"] = items
            calc_no, plan_no = _save_pack_result(result_id, agent_result,
                                                  input_bins=user_bins or bins,
                                                  session_id=sid,
                                                  plan_type=mat_type,
                                                  classify_result=effective_classify,
                                                  duration_ms=_duration_ms,
                                                  classify_input_tokens=_classify_in_tok,
                                                  classify_output_tokens=_classify_out_tok,
                                                  classify_source=_classify_source,
                                                  classify_model=_classify_model,
                                                  classify_provider=_classify_provider)
            agent_result["calc_no"] = calc_no
            agent_result["plan_no"] = plan_no

            # ── 计算结果异常（在 calc_no 生成之后记录）────────────────────────
            _fr = agent_result["final_result"].get("summary", {})
            _util   = _fr.get("avg_utilization", 1.0)
            _placed = _fr.get("all_placed", True)
            if not _placed:
                log_anomaly(
                    anomaly_type="calc_anomaly", severity="error",
                    session_id=sid, calc_no=calc_no,
                    error_code="unplaced_items",
                    extra={"utilization": round(_util, 3), "item_count": len(items)},
                )
            elif _util < 0.3:
                log_anomaly(
                    anomaly_type="calc_anomaly", severity="warning",
                    session_id=sid, calc_no=calc_no,
                    error_code="low_utilization",
                    extra={"utilization": round(_util, 3), "item_count": len(items)},
                )

        # ── AI 调用失败（在 calc_no 生成之后记录）────────────────────────────
        _ai_error = agent_result.get("ai_error")
        if _ai_error:
            log_anomaly(
                anomaly_type="ai_failure",
                severity="critical",
                session_id=sid,
                calc_no=calc_no,
                error_msg=_ai_error,
                extra={"item_count": len(items)},
            )

        # ── 单次超时（> 30s）────────────────────────────────────────────────
        if _duration_ms > _AGENT_PACK_SINGLE_TIMEOUT_MS:
            log_anomaly(
                anomaly_type="calc_timeout",
                severity="warning",
                session_id=sid,
                calc_no=calc_no,
                duration_ms=_duration_ms,
                extra={"item_count": len(items), "scope": "single"},
            )

        # ── Session 累计超时（> 90s）─────────────────────────────────────────
        if sid and sid in _session_timing:
            _session_elapsed_ms = int((_time.time() - _session_timing[sid]["start_ts"]) * 1000)
            _call_count = _session_timing[sid]["call_count"]
            if _session_elapsed_ms > _AGENT_PACK_SESSION_TIMEOUT_MS:
                log_anomaly(
                    anomaly_type="calc_timeout",
                    severity="warning",
                    session_id=sid,
                    calc_no=calc_no,
                    duration_ms=_session_elapsed_ms,
                    extra={"item_count": len(items), "scope": "session",
                           "call_count": _call_count},
                )
                # 记录后清除，避免后续每次请求都重复触发
                _session_timing.pop(sid, None)

        # 将推荐包材类型注入 compare_summary（供前端对比面板和反馈使用）
        if mat_type and agent_result.get("compare_result"):
            cs = agent_result["compare_result"].get("compare_summary")
            if isinstance(cs, dict):
                cs["mat_type"] = mat_type
        # 顶层也暴露，便于前端 resp.plan_type 直接读取
        agent_result["plan_type"] = mat_type

        # ── ai_error 脱敏：内部错误不透传给前端，避免暴露技术细节 ──────────────
        if agent_result.get("ai_error"):
            agent_result["ai_error"] = "AI调用失败，已降级为本地算法"

        # ── 响应数据脱敏：移除 compare_result / final_result 中的产品敏感字段 ──
        # 前端 3D 渲染和对比面板不依赖 product_title / sale_price 等字段
        if agent_result.get("compare_result"):
            cr = agent_result["compare_result"]
            cr.pop("_soft_ai_meta", None)
            for key in ("recommended_result", "best_full_result", "soft_full_result"):
                if cr.get(key):
                    cr[key] = _clean_packed_result(cr[key])
            # top3_full_results：逐一脱敏
            if isinstance(cr.get("top3_full_results"), list):
                cr["top3_full_results"] = [
                    _clean_packed_result(r) for r in cr["top3_full_results"]
                ]
            # compare_summary：清理 top3_existing[].bin / recommended_bin / soft_bin
            if isinstance(cr.get("compare_summary"), dict):
                cr["compare_summary"] = _sanitize_compare_summary(cr["compare_summary"])
        if agent_result.get("final_result"):
            agent_result["final_result"] = _clean_packed_result(agent_result["final_result"])
        # items 字段（存档用途）也做脱敏后去除，不返回给前端
        agent_result.pop("items", None)

        return agent_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    """查询已保存的装箱结果（供可视化页面调用）"""
    result = _load_result(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="结果不存在或已过期")
    return result


# ── 复现数据脱敏 ─────────────────────────────────────────────────────────────

# 货物记录中的敏感字段：展示层不需要，不出前端
_ITEM_SENSITIVE_FIELDS = frozenset({
    "product_title", "sale_price", "product_category",
    "soft_packaging_ok", "product_name",
})

# 装箱结果中的内部调试字段
_RESULT_INTERNAL_FIELDS = frozenset({"_soft_ai_meta"})

# classify_result 只保留展示所需字段，其余（token 数、模型名等）不出前端
_CLASSIFY_DISPLAY_FIELDS = frozenset({"protection_level", "recommended_types", "reason"})


def _clean_packed_items(items_list: list) -> list:
    """移除 packed_bins[].items[] 中的产品敏感字段，保留 3D 渲染所需字段"""
    if not isinstance(items_list, list):
        return items_list
    return [
        {k: v for k, v in item.items() if k not in _ITEM_SENSITIVE_FIELDS}
        if isinstance(item, dict) else item
        for item in items_list
    ]


def _clean_packed_result(result: dict | None) -> dict | None:
    """清理单个装箱结果：去掉货物敏感信息和内部调试字段"""
    if not isinstance(result, dict):
        return result
    r = {k: v for k, v in result.items() if k not in _RESULT_INTERNAL_FIELDS}
    if "packed_bins" in r:
        cleaned = []
        for b in (r["packed_bins"] or []):
            if isinstance(b, dict):
                b = dict(b)
                if "items" in b:
                    b["items"] = _clean_packed_items(b["items"])
            cleaned.append(b)
        r["packed_bins"] = cleaned
    return r


# recommended_bin / soft_bin 前端展示所需字段（其他字段不出前端）
_BIN_DISPLAY_FIELDS = frozenset({
    "type", "length", "width", "height", "max_weight", "sku", "cost_price",
})


def _sanitize_compare_summary(cs: dict | None) -> dict | None:
    """
    清理 compare_summary：
    - top3_existing[].bin  → 删除（完整目录记录含供应商价格，前端未使用）
    - recommended_bin      → 只保留展示字段
    - soft_bin             → 只保留展示字段
    """
    if not isinstance(cs, dict):
        return cs
    cs = dict(cs)

    # top3_existing：删除每条里的 bin 字段（全量目录数据）
    if isinstance(cs.get("top3_existing"), list):
        cs["top3_existing"] = [
            {k: v for k, v in entry.items() if k != "bin"}
            if isinstance(entry, dict) else entry
            for entry in cs["top3_existing"]
        ]

    # recommended_bin / soft_bin：只保留前端展示所需字段
    for key in ("recommended_bin", "soft_bin"):
        if isinstance(cs.get(key), dict):
            cs[key] = {k: v for k, v in cs[key].items() if k in _BIN_DISPLAY_FIELDS}

    return cs


def _sanitize_replay_plan(plan: dict) -> dict:
    """
    复现方案脱敏：只保留前端展示所需数据，去掉产品明细和内部字段。
    - packed_bins[].items[] 移除 product_title / sale_price 等产品信息
    - classify_result 只保留 protection_level / recommended_types / reason
    - 移除 _soft_ai_meta 等内部调试字段
    """
    d = dict(plan)

    # final_result
    if d.get("final_result"):
        d["final_result"] = _clean_packed_result(d["final_result"])

    # compare_result
    if isinstance(d.get("compare_result"), dict):
        cr = {k: v for k, v in d["compare_result"].items() if k not in _RESULT_INTERNAL_FIELDS}
        # 单个装箱结果字段
        for key in ("recommended_result", "best_full_result", "soft_full_result"):
            if cr.get(key):
                cr[key] = _clean_packed_result(cr[key])
        # top3_full_results：数组，每个元素都是完整装箱结果，逐一脱敏
        if isinstance(cr.get("top3_full_results"), list):
            cr["top3_full_results"] = [
                _clean_packed_result(r) for r in cr["top3_full_results"]
            ]
        # compare_summary：清理 top3_existing[].bin / recommended_bin / soft_bin
        if isinstance(cr.get("compare_summary"), dict):
            cr["compare_summary"] = _sanitize_compare_summary(cr["compare_summary"])
        d["compare_result"] = cr

    # classify_result — 只留展示字段
    if isinstance(d.get("classify_result"), dict):
        d["classify_result"] = {
            k: v for k, v in d["classify_result"].items()
            if k in _CLASSIFY_DISPLAY_FIELDS
        }

    return d


@app.get("/api/replay/{calc_no}")
async def replay_scheme(calc_no: str):
    """通过计算编号复现历史方案存档"""
    conn = _get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT calc_no, plan_no, session_id, plan_type, classify_result, agent_summary, "
                "final_result, compare_result, created_at "
                "FROM pack_scheme_detail WHERE calc_no=%s ORDER BY plan_no",
                (calc_no,)
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="未找到该计算编号的存档，仅支持复现新版本记录的方案")

    plans = []
    for r in rows:
        d = dict(r)
        d["final_result"]    = json.loads(d["final_result"])    if d["final_result"]    else None
        d["compare_result"]  = json.loads(d["compare_result"])  if d["compare_result"]  else None
        d["classify_result"] = json.loads(d["classify_result"]) if isinstance(d.get("classify_result"), str) else d.get("classify_result")
        if isinstance(d.get("created_at"), datetime.datetime):
            d["created_at"] = d["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        # 脱敏处理：移除产品明细和内部调试字段，只保留展示数据
        plans.append(_sanitize_replay_plan(d))

    return {"calc_no": calc_no, "plans": plans}


@app.post("/api/recognize-shape")
async def recognize_shape(request: RecognizeRequest):
    """调用 Claude Vision 识别产品形状和尺寸（支持多角度照片）"""
    try:
        _client = anthropic.Anthropic(timeout=60.0)

        # 构建多图内容：每张图片后附上角度说明
        content = []
        for img in request.images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.image_base64,
                },
            })
            content.append({
                "type": "text",
                "text": f"（以上是产品的{img.view}照片）",
            })

        view_desc = "、".join(img.view for img in request.images)
        content.append({
            "type": "text",
            "text": (
                f"以上是同一产品的{view_desc}照片，请综合多角度分析产品的形状、尺寸和品类，"
                "仅返回如下JSON（不要任何其他文字）：\n"
                '{"shape_type":"box","length":30,"width":20,"height":15,'
                '"confidence":0.85,"shape_note":"简短描述",'
                '"product_category":"手机","fragility_level":"high"}\n\n'
                "shape_type 只能是：box(长方体)、cylinder(圆柱体,length=width=直径)、"
                "sphere(球体,三边相等)、cone(锥形)、l_shape(L形)、irregular(其他异形)\n"
                "尺寸单位cm，length最长边，width次之，height最短。\n"
                "product_category：产品品类中文名称，如「手机」「陶瓷」「服装」等。\n"
                "fragility_level：脆弱等级，只能是 high（易碎/精密）/ medium（一般）/ low（耐摔）。\n"
                "若图中有参照物（A4纸/尺子/硬币等）请借助推算；"
                "否则凭产品外观常识估算，confidence设低。"
            ),
        })

        response = _client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            messages=[{"role": "user", "content": content}],
        )
        text = response.content[0].text.strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        result = json.loads(m.group() if m else text)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/embed-provider")
async def api_get_embed_provider():
    """返回当前 embedding provider 及各集合的辅材条数"""
    provider = get_provider()
    return {
        "provider": provider,
        "counts": {
            "voyage": count_materials("voyage"),
            "local":  count_materials("local"),
        },
    }


@app.post("/api/embed-provider")
async def api_set_embed_provider(body: dict):
    """切换 embedding provider（voyage / local）"""
    p = body.get("provider", "")
    try:
        set_provider(p)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "provider": p,
        "count": count_materials(p),
    }


@app.post("/api/smart-pack")
async def smart_pack(request: SmartPackRequest):
    """智能包装：向量检索辅材 + Claude 精排 + 最优外箱 + 利润率分析"""
    try:
        # 1. 构建向量检索 query
        fragility_desc = {
            "high":   "高脆弱易碎 防震缓冲 内衬保护",
            "medium": "中等脆弱 基础防护 填充缓冲",
            "low":    "耐摔 轻量 简单填充",
        }.get(request.fragility_level, "基础防护")
        query = f"{request.product_category} {fragility_desc} 包装辅材"

        # 2. 向量检索 Top20 候选
        candidates = search_materials(query, top_k=20)
        candidate_text = "\n".join(
            f"[{i+1}] SKU:{c['sku']} | 品类:{c['category']} | 名称:{c['name']} "
            f"| 价格:¥{c['price']} | 尺寸:{c['size']} | 相似度:{c['similarity']}"
            for i, c in enumerate(candidates)
        )

        # 3. Claude 精排：从候选中选 3-5 种，估算每种单侧填充厚度
        _client = anthropic.Anthropic(timeout=30.0)
        prompt = (
            f"你是专业包装工程师。根据产品信息，从候选辅材中选出最合适的3-5种。\n\n"
            f"产品信息：\n"
            f"- 品类：{request.product_category}\n"
            f"- 尺寸：{request.length} × {request.width} × {request.height} cm\n"
            f"- 重量：{request.weight} kg\n"
            f"- 脆弱等级：{request.fragility_level}（high=易碎，medium=普通，low=耐摔）\n\n"
            f"候选辅材：\n{candidate_text}\n\n"
            f"请以JSON数组返回所选辅材（只输出JSON，不要其他文字）：\n"
            f'[{{"sku":"SKU","name":"名称","price":0.0,'
            f'"padding_each_side_mm":10,"description":"选择理由"}}]'
        )
        resp = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = resp.content[0].text.strip()
        # 提取 JSON 数组（防止 Claude 在前后加文字）
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        materials_raw = json.loads(match.group(0) if match else raw_text)

        # 4. 计算加辅材后的实装尺寸
        total_pad_cm = max((m.get("padding_each_side_mm", 0) for m in materials_raw), default=0) / 10
        packed_dims = {
            "length": round(request.length + total_pad_cm * 2, 2),
            "width":  round(request.width  + total_pad_cm * 2, 2),
            "height": round(request.height + total_pad_cm * 2, 2),
        }

        # 5. 辅材总成本
        material_cost = round(sum(float(m.get("price", 0)) for m in materials_raw), 2)

        # 6. 构建装箱物品
        items = [
            {
                "id":     f"item_{i+1}",
                "length": packed_dims["length"],
                "width":  packed_dims["width"],
                "height": packed_dims["height"],
                "weight": request.weight,
            }
            for i in range(request.quantity)
        ]

        # 7. 调用装箱引擎（优先用推荐箱型，再考虑现有箱型）
        rec_bin = _calc_recommended_bin(items)
        packing_result = calculate_packing(items, AVAILABLE_BINS + [rec_bin])

        # 8. 利润率计算
        product_volume = request.length * request.width * request.height
        box_volume = 0.0
        if packing_result["packed_bins"]:
            b = packing_result["packed_bins"][0]["dimensions"]
            box_volume = b["length"] * b["width"] * b["height"]

        space_utilization = packing_result["packed_bins"][0]["utilization"] if packing_result["packed_bins"] else 0.0
        cost_per_cm3 = round(material_cost / product_volume, 4) if product_volume > 0 else 0.0
        cost_normalized = min(cost_per_cm3 / 0.1, 1.0)
        efficiency_score = round(min(space_utilization, 1.0) * 60 + (1 - cost_normalized) * 40)

        profit_analysis = {
            "product_volume_cm3": round(product_volume, 2),
            "box_volume_cm3":     round(box_volume, 2),
            "space_utilization":  space_utilization,
            "material_cost_yuan": material_cost,
            "cost_per_cm3":       cost_per_cm3,
            "efficiency_score":   efficiency_score,
        }

        # 前端兼容格式：type + unit_price + description
        materials_out = [
            {
                "type":        m.get("name", ""),
                "sku":         m.get("sku", ""),
                "unit_price":  float(m.get("price", 0)),
                "description": m.get("description", ""),
                "padding_mm":  m.get("padding_each_side_mm", 0),
            }
            for m in materials_raw
        ]

        return {
            "success":            True,
            "fragility_used":     request.fragility_level,
            "materials":          materials_out,
            "material_cost_total": material_cost,
            "packed_dims":        packed_dims,
            "packing_result":     packing_result,
            "profit_analysis":    profit_analysis,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-3d")
async def generate_3d(request: Generate3DRequest):
    """TripoSR 本地从单图生成 GLB 3D 模型"""
    try:
        from triposr_worker import generate_glb
        loop = asyncio.get_event_loop()
        glb_bytes = await loop.run_in_executor(
            None, generate_glb, request.image_base64, request.media_type
        )
        glb_b64 = base64.b64encode(glb_bytes).decode()
        return {"success": True, "glb_base64": glb_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_TRIPO_BASE = "https://api.tripo3d.ai/v2/openapi"

@app.post("/api/tripo3d/generate")
async def tripo3d_generate(request: Tripo3DRequest):
    """调用 Tripo3D API 从单图生成 GLB 3D 模型"""
    # API Key 由服务端 .env 统一管理（TRIPO3D_API_KEY），前端不传
    _tripo_key = os.environ.get("TRIPO3D_API_KEY", "")
    if not _tripo_key:
        raise HTTPException(status_code=503, detail="Tripo3D API Key 未配置，请联系管理员在服务器 .env 中设置 TRIPO3D_API_KEY")
    headers = {"Authorization": f"Bearer {_tripo_key}"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 1: 上传图片
        img_bytes = base64.b64decode(request.image_base64)
        ext = "jpg" if "jpeg" in request.media_type else request.media_type.split("/")[-1]
        files = {"file": (f"image.{ext}", img_bytes, request.media_type)}
        resp = await client.post(f"{_TRIPO_BASE}/upload", headers=headers, files=files)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code,
                                detail=f"上传失败: {resp.text}")
        upload_data = resp.json()
        if upload_data.get("code") != 0:
            raise HTTPException(status_code=400,
                                detail=f"上传失败: {upload_data.get('message', resp.text)}")
        image_token = upload_data["data"]["image_token"]

        # Step 2: 创建 image_to_model 任务
        resp = await client.post(
            f"{_TRIPO_BASE}/task",
            headers={**headers, "Content-Type": "application/json"},
            json={"type": "image_to_model", "file": {"type": "jpg", "file_token": image_token}},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code,
                                detail=f"创建任务失败: {resp.text}")
        task_data = resp.json()
        if task_data.get("code") != 0:
            raise HTTPException(status_code=400,
                                detail=f"创建任务失败: {task_data.get('message', resp.text)}")
        task_id = task_data["data"]["task_id"]

        # Step 3: 轮询任务状态（最多等 120 秒）
        for _ in range(40):
            await asyncio.sleep(3)
            resp = await client.get(f"{_TRIPO_BASE}/task/{task_id}", headers=headers)
            if resp.status_code != 200:
                continue
            status_data = resp.json()
            if status_data.get("code") != 0:
                continue
            task_info = status_data["data"]
            status = task_info.get("status", "")
            if status == "success":
                glb_url = task_info.get("output", {}).get("model", "")
                break
            elif status in ("failed", "cancelled", "unknown"):
                raise HTTPException(status_code=500,
                                    detail=f"Tripo3D 任务失败: {status}")
        else:
            raise HTTPException(status_code=504, detail="Tripo3D 任务超时（>120秒）")

        # Step 4: 下载 GLB 文件
        resp = await client.get(glb_url)
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="GLB 下载失败")
        glb_b64 = base64.b64encode(resp.content).decode()

    return {"success": True, "glb_base64": glb_b64, "task_id": task_id}


@app.get("/view")
async def view(id: str):
    """返回 Three.js 3D 可视化页面"""
    if _load_result(id) is None:
        raise HTTPException(status_code=404, detail="结果不存在或已过期")
    return FileResponse(os.path.join(STATIC_DIR, "viewer.html"))


@app.get("/")
async def index():
    """前端装箱系统主页"""
    return FileResponse(
        os.path.join(STATIC_DIR, "app.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/catalog-status")
async def catalog_status():
    """诊断接口：确认包材目录是否加载、预筛选是否工作"""
    from agent import _RAW_CATALOG, _prefilter_catalog_bins, AVAILABLE_BINS
    test_items = [
        {"id": "A", "length": 45, "width": 15, "height": 5,  "weight": 0.5},
        {"id": "B", "length": 12, "width": 8,  "height": 4,  "weight": 0.3},
    ]
    filtered = _prefilter_catalog_bins(test_items)
    using_catalog = filtered is not AVAILABLE_BINS and len(filtered) > 0 and filtered[0].get("sku", "") != ""
    return {
        "catalog_loaded":   len(_RAW_CATALOG),
        "prefilter_count":  len(filtered),
        "using_catalog":    using_catalog,
        "top3_candidates":  [
            {"type": b["type"], "dims": f"{b['length']}x{b['width']}x{b['height']}",
             "cost": b.get("cost_price"), "sku": b.get("sku")}
            for b in filtered[:3]
        ],
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
