"""
TripoSR 本地 3D 生成服务
背景去除策略（优先级）：
  1. rembg + u2net（需 ~/.u2net/u2net.onnx，效果最好）
  2. OpenCV GrabCut（免下载，效果良好）
"""
import os
import sys
import base64
import io
import tempfile

import numpy as np
import cv2
import torch
from PIL import Image

# TripoSR 目录
TRIPOSR_DIR = os.path.join(os.path.dirname(__file__), "..", "TripoSR")
if TRIPOSR_DIR not in sys.path:
    sys.path.insert(0, TRIPOSR_DIR)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tsr.system import TSR
from tsr.utils import resize_foreground

# 单例
_model    = None
_device   = None
_rembg_session = None

_U2NET_PATH = os.path.join(os.path.expanduser("~"), ".u2net", "u2net.onnx")


def _u2net_available() -> bool:
    return os.path.isfile(_U2NET_PATH) and os.path.getsize(_U2NET_PATH) > 1_000_000


def _get_model_path() -> str:
    def _has_model(p):
        return (os.path.isfile(os.path.join(p, "config.yaml")) and
                os.path.isfile(os.path.join(p, "model.ckpt")))

    ms = os.path.join(os.path.expanduser("~"), ".cache", "modelscope",
                      "AI-ModelScope", "TripoSR")
    if _has_model(ms):
        return ms

    hf_base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                           "hub", "models--stabilityai--TripoSR", "snapshots")
    if os.path.isdir(hf_base):
        for snap in os.listdir(hf_base):
            p = os.path.join(hf_base, snap)
            if _has_model(p):
                return p

    return "stabilityai/TripoSR"


# ── 背景去除：GrabCut ──────────────────────────────────────────────────────

def _grabcut_remove_bg(pil_img: Image.Image) -> Image.Image:
    """
    用 OpenCV GrabCut 去除背景，返回 RGBA 图像。
    假设产品位于图像中央区域（适合产品拍摄场景）。
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    h, w    = img_rgb.shape[:2]

    # 前景矩形：中央 70% 区域
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask   = np.zeros((h, w), np.uint8)
    bgd    = np.zeros((1, 65), np.float64)
    fgd    = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_rgb, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    # 0=确定背景, 2=可能背景 → 0；1=确定前景, 3=可能前景 → 1
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

    # 形态学：闭运算填充空洞，膨胀轻微扩大前景
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE,  kernel, iterations=3)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    # 高斯模糊软化边缘
    alpha = (cv2.GaussianBlur(fg_mask * 255, (21, 21), 0)).astype(np.uint8)

    rgba = np.dstack([img_rgb, alpha])
    return Image.fromarray(rgba, "RGBA")


# ── 图像预处理 ─────────────────────────────────────────────────────────────

def _preprocess(pil_img: Image.Image) -> Image.Image:
    """去背 → 合成灰底 → 归一化"""
    if _u2net_available():
        import rembg
        from tsr.utils import remove_background
        global _rembg_session
        if _rembg_session is None:
            _rembg_session = rembg.new_session()
        rgba = remove_background(pil_img.convert("RGBA"), _rembg_session)
        print("[TripoSR] 使用 rembg 背景去除")
    else:
        rgba = _grabcut_remove_bg(pil_img)
        print("[TripoSR] 使用 GrabCut 背景去除")

    rgba   = resize_foreground(rgba, 0.85)
    arr    = np.array(rgba).astype(np.float32) / 255.0
    # alpha 合成灰底 (0.5, 0.5, 0.5)
    arr    = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((arr * 255.0).astype(np.uint8))


# ── 模型加载 ───────────────────────────────────────────────────────────────

def _get_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    _device     = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path  = _get_model_path()
    print(f"[TripoSR] 加载模型: {model_path}  设备: {_device}")

    _model = TSR.from_pretrained(
        model_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    _model.renderer.set_chunk_size(4096)  # 降低峰值显存占用
    _model.to(_device)
    print("[TripoSR] 模型就绪")
    return _model, _device


# ── 主接口 ────────────────────────────────────────────────────────────────

def generate_glb(image_base64: str, media_type: str = "image/jpeg") -> bytes:
    model, device = _get_model()

    img_bytes = base64.b64decode(image_base64)
    pil_img   = Image.open(io.BytesIO(img_bytes))
    processed = _preprocess(pil_img)

    # 推理前释放显存碎片
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    with torch.no_grad():
        scene_codes = model([processed], device=device)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=256)

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        meshes[0].export(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
