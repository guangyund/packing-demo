"""
向量数据库模块 - ChromaDB 封装
负责辅材数据的写入和相似度检索
"""
import os
import chromadb
from embedder import embed_documents, embed_query, get_provider

# ChromaDB 数据持久化目录
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def _get_collection(provider: str = None):
    """获取 ChromaDB 集合；provider 为 None 时使用当前全局 provider"""
    p = provider or get_provider()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=f"materials_{p}",
        metadata={"hnsw:space": "cosine"},  # 余弦相似度
    )


def upsert_materials(materials: list[dict]):
    """
    写入/更新辅材向量。
    materials 每项需包含：sku, name, price, size, weight, category
    """
    collection = _get_collection()

    # 用名称 + 品类拼接成检索文本，信息越丰富效果越好
    texts = [
        f"{m['category']} {m['name']}"
        for m in materials
    ]
    embeddings = embed_documents(texts)

    collection.upsert(
        ids=[m["sku"] for m in materials],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "sku":      m["sku"],
            "name":     m["name"],
            "price":    float(m.get("price") or 0),
            "size":     m.get("size", ""),
            "weight":   float(m.get("weight") or 0),
            "category": m.get("category", ""),
        } for m in materials],
    )
    return len(materials)


def search_materials(query: str, top_k: int = 20, category_filter: str = None) -> list[dict]:
    """
    向量检索最相似的辅材。
    query:           需求描述文字
    top_k:           返回条数
    category_filter: 只搜某个大类（可选）
    """
    collection = _get_collection()
    query_embedding = embed_query(query)

    where = {"category": category_filter} if category_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["metadatas", "distances", "documents"],
    )

    items = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        items.append({
            **meta,
            "similarity": round(1 - dist, 4),  # 距离转相似度
        })
    return items


def count_materials(provider: str = None) -> int:
    """统计指定 provider 集合的条数；provider 为 None 时用当前全局 provider"""
    return _get_collection(provider).count()
