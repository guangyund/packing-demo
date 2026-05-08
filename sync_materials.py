"""
辅材数据同步脚本 - 从 Excel 导入到 ChromaDB
后续接入 MySQL 时只需替换 load_from_excel() 为 load_from_mysql()
"""
import os
import sys
import io
import openpyxl

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "耗材.xlsx")


def load_from_excel(path: str = EXCEL_PATH) -> list[dict]:
    """从 Excel 读取辅材数据"""
    wb = openpyxl.load_workbook(path)
    ws = wb.active

    materials = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        sku, name, price, size, weight, vol_weight, category = row[:7]
        if not sku or not name:
            continue
        materials.append({
            "sku":      str(sku).strip(),
            "name":     str(name).strip(),
            "price":    price or 0,
            "size":     str(size).strip() if size else "",
            "weight":   weight or 0,
            "category": str(category).strip() if category else "",
        })
    return materials


def sync(batch_size: int = 30):
    """全量同步 Excel → ChromaDB（支持断点续传，跳过已存在的 SKU）"""
    from vector_store import upsert_materials, count_materials, _get_collection

    print("读取 Excel 数据...")
    materials = load_from_excel()
    print(f"共 {len(materials)} 条辅材记录")

    # 查询已存在的 SKU，跳过已处理的
    collection = _get_collection()
    existing = set(collection.get(include=[])["ids"])
    pending = [m for m in materials if m["sku"] not in existing]
    print(f"已写入 {len(existing)} 条，剩余待处理 {len(pending)} 条")

    if not pending:
        print("全部已同步完成")
        return

    import time
    import voyageai

    def upsert_with_retry(batch, max_retries=5):
        for attempt in range(max_retries):
            try:
                upsert_materials(batch)
                return
            except voyageai.error.RateLimitError:
                wait = 60 * (attempt + 1)
                print(f"  触发速率限制，等待 {wait} 秒后重试...")
                time.sleep(wait)
        raise RuntimeError("超过最大重试次数")

    total = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i+batch_size]
        upsert_with_retry(batch)
        total += len(batch)
        print(f"  已处理 {len(existing)+total}/{len(materials)} 条...")
        if i + batch_size < len(pending):
            time.sleep(21)  # 免费账号 3 RPM，每批间隔 21 秒

    print(f"同步完成，ChromaDB 共 {count_materials()} 条")


if __name__ == "__main__":
    sync()
