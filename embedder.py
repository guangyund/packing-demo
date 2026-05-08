"""
Embedding 模块 - 支持 Voyage API 和本地模型两种方式，通过 PROVIDER 切换
"""
import os

# 运行时可通过 set_provider() 动态切换，初始值读环境变量
_provider: str = os.environ.get("EMBED_PROVIDER", "voyage")
# 保持向后兼容（旧代码直接引用 PROVIDER 的地方不报错）
PROVIDER = _provider


def get_provider() -> str:
    return _provider


def set_provider(p: str) -> None:
    global _provider, PROVIDER
    if p not in ("voyage", "local"):
        raise ValueError(f"Unknown provider '{p}', must be 'voyage' or 'local'")
    _provider = p
    PROVIDER = p

# ── Voyage API ────────────────────────────────────────────────────────────────
def _voyage_embed(texts: list[str]) -> list[list[float]]:
    import voyageai
    client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY", ""))
    result = client.embed(texts, model="voyage-3", input_type="document")
    return result.embeddings


def _voyage_embed_query(text: str) -> list[float]:
    import voyageai
    client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY", ""))
    result = client.embed([text], model="voyage-3", input_type="query")
    return result.embeddings[0]


# ── 本地模型（FlagEmbedding bge-m3）──────────────────────────────────────────
def _local_embed(texts: list[str]) -> list[list[float]]:
    from FlagEmbedding import BGEM3FlagModel
    if not hasattr(_local_embed, "_model"):
        _local_embed._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    result = _local_embed._model.encode(texts, batch_size=32, max_length=512)
    return result["dense_vecs"].tolist()


def _local_embed_query(text: str) -> list[float]:
    return _local_embed([text])[0]


# ── 统一接口 ──────────────────────────────────────────────────────────────────
def embed_documents(texts: list[str]) -> list[list[float]]:
    """批量生成文档 embedding（用于建库）"""
    if _provider == "voyage":
        import time
        results = []
        batch = 30  # 免费账号限 10K TPM，每条约100token，30条≈3000token留余量
        for i in range(0, len(texts), batch):
            results.extend(_voyage_embed(texts[i:i+batch]))
            if i + batch < len(texts):
                time.sleep(21)  # 免费账号 3 RPM，等21秒确保不超限
        return results
    else:
        return _local_embed(texts)


def embed_query(text: str) -> list[float]:
    """生成查询 embedding（用于检索）"""
    if _provider == "voyage":
        return _voyage_embed_query(text)
    else:
        return _local_embed_query(text)
