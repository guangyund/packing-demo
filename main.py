"""
三维装箱服务 - FastAPI Web 服务
接口：
  POST /api/pack          直接装箱（手动指定箱型）
  POST /api/agent-pack    AI Agent 自动选箱并计算
  GET  /api/results/{id}  查询装箱结果
  GET  /view              Three.js 3D 可视化页面
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 加载 .env（VOYAGE_API_KEY、ANTHROPIC_API_KEY 等）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

import uuid
import re
import json
import base64
import asyncio
import httpx
import anthropic
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from packing_engine import calculate_packing, AVAILABLE_BINS
from agent import run_packing_agent, _calc_recommended_bin

# ── 包材目录（从 Excel 导出的 JSON）─────────────────────────────────────────────
import json as _json
_BINS_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "bins_catalog.json")
try:
    with open(_BINS_CATALOG_PATH, encoding="utf-8") as _f:
        _BINS_CATALOG: list = _json.load(_f)
except FileNotFoundError:
    _BINS_CATALOG = []
from material_advisor import (
    get_fragility, recommend_materials,
    calc_packed_dimensions, calc_material_cost,
)
from vector_store import search_materials, count_materials
from embedder import get_provider, set_provider

app = FastAPI(title="3D Bin Packing Service", version="1.0.0")

# 静态文件目录（viewer.html）
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 内存结果存储（生产环境建议换成 Redis / 数据库）
_results: dict = {}


# ── 数据模型 ──────────────────────────────────────────────────────────────────

class ItemModel(BaseModel):
    id: str
    length: float
    width: float
    height: float
    weight: float
    sale_price: float = 0.0             # 产品售价（USD），用于运费档级计算
    product_category: str = "常规类产品"  # 常规类产品 / 服装产品 / 危险品
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
    api_key: str


class SmartPackRequest(BaseModel):
    product_category: str
    fragility_level: str           # high / medium / low
    length: float                  # 产品净尺寸 cm
    width: float
    height: float
    weight: float                  # kg
    quantity: int = 1


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _save_result(result: dict) -> str:
    """保存装箱结果并返回唯一 ID"""
    result_id = str(uuid.uuid4())
    _results[result_id] = result
    return result_id


# ── 接口 ──────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/bins-catalog")
async def bins_catalog(
    q:    str = "",
    type: str = "",
    page: int = 1,
    page_size: int = 50,
):
    """
    查询包材目录。
    q        — 关键词搜索（SKU / 名称 / 包材名称）
    type     — 过滤类型：硬包材 / 软包材 / 空字符串=全部
    page     — 页码（从 1 开始）
    page_size — 每页条数（最多 200）
    """
    page_size = min(page_size, 200)
    data = _BINS_CATALOG
    if type in ("硬包材", "软包材"):
        data = [c for c in data if c["type"] == type]
    if q:
        ql = q.lower()
        data = [c for c in data if
                ql in c["sku"].lower() or
                ql in c["name"].lower() or
                ql in c["mat_name"].lower()]
    total = len(data)
    start = (page - 1) * page_size
    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "items":     data[start: start + page_size],
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


@app.post("/api/agent-pack")
async def agent_pack(request: AgentPackRequest):
    """AI Agent 自动选箱并计算装箱方案"""
    try:
        items = [item.model_dump() for item in request.items]
        bins  = [b.model_dump() for b in request.bins] if request.bins else None
        agent_result = run_packing_agent(items, bins=bins)

        if agent_result.get("success") and agent_result.get("final_result"):
            result_id = _save_result(agent_result["final_result"])
            agent_result["result_id"] = result_id
            agent_result["view_url"] = f"/view?id={result_id}"

        return agent_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    """查询已保存的装箱结果（供可视化页面调用）"""
    if result_id not in _results:
        raise HTTPException(status_code=404, detail="结果不存在或已过期")
    return _results[result_id]


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
    headers = {"Authorization": f"Bearer {request.api_key}"}

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
    if id not in _results:
        raise HTTPException(status_code=404, detail="结果不存在或已过期")
    return FileResponse(os.path.join(STATIC_DIR, "viewer.html"))


@app.get("/")
async def index():
    """前端装箱系统主页"""
    return FileResponse(os.path.join(STATIC_DIR, "app.html"))


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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
