"""
本地测试脚本 - 不需要启动服务器，直接运行
用法：python test_local.py
"""
import json
import os
from packing_engine import calculate_packing
from agent import run_packing_agent
from visualizer import visualize_packing

# ── 测试数据：模拟一张出库订单 ────────────────────────────────────────────────
TEST_ITEMS = [
    {"id": "SKU001-键盘",   "length": 45, "width": 15, "height":  5, "weight": 0.8},
    {"id": "SKU002-鼠标",   "length": 12, "width":  8, "height":  4, "weight": 0.2},
    {"id": "SKU003-显示器", "length": 55, "width": 35, "height": 15, "weight": 4.5},
    {"id": "SKU004-耳机",   "length": 20, "width": 18, "height": 10, "weight": 0.3},
    {"id": "SKU005-充电器", "length": 10, "width":  6, "height":  4, "weight": 0.2},
]


def print_result(result: dict):
    """格式化打印装箱结果"""
    summary = result.get("summary", {})
    print(f"  使用箱子数：{summary.get('total_bins_used', 0)}")
    print(f"  平均利用率：{summary.get('avg_utilization', 0) * 100:.1f}%")
    print(f"  全部装入：  {'是' if summary.get('all_placed') else '否'}")

    if result.get("unplaced_items"):
        print(f"  未装入货物：{result['unplaced_items']}")

    for i, b in enumerate(result.get("packed_bins", []), 1):
        print(f"\n  箱子 {i}：{b['bin_type']}")
        print(f"    利用率：{b['utilization'] * 100:.1f}%  "
              f"总重：{b['total_weight']}kg  货物数：{b['item_count']}")
        for item in b["items"]:
            p = item["position"]
            d = item["dimensions"]
            print(f"    - {item['id']:<20} "
                  f"位置({p['x']:.0f},{p['y']:.0f},{p['z']:.0f})  "
                  f"尺寸{d['length']:.0f}x{d['width']:.0f}x{d['height']:.0f}")


def test_direct():
    """测试一：直接装箱算法"""
    print("\n" + "=" * 60)
    print("【测试一】直接装箱计算 + matplotlib 3D可视化")
    print("=" * 60)

    bins = [
        {"type": "中号箱", "length": 60, "width": 50, "height": 50, "max_weight": 30},
    ]

    result = calculate_packing(TEST_ITEMS, bins)
    print_result(result)

    print("\n[正在打开3D可视化窗口...]")
    visualize_packing(result, save_path="packing_result.png")
    print("[已保存至 packing_result.png，同时弹窗显示]")


def test_agent():
    """测试二：AI Agent 自动装箱"""
    print("\n" + "=" * 60)
    print("【测试二】AI Agent 自动选箱 + matplotlib 3D可视化")
    print("=" * 60)

    result = run_packing_agent(TEST_ITEMS)

    print(f"\n[Agent 分析]\n{result['agent_summary']}")

    if result.get("final_result"):
        print("\n[最终装箱方案]")
        print_result(result["final_result"])

        print("\n[正在打开3D可视化窗口...]")
        visualize_packing(result["final_result"], save_path="agent_result.png")
        print("[已保存至 agent_result.png]")
    else:
        print("[错误]", result.get("error"))


if __name__ == "__main__":
    print("三维装箱 Demo 测试")
    print("货物清单：", [item["id"] for item in TEST_ITEMS])

    # 测试一：直接算法（无需 API Key）
    test_direct()

    # 测试二：AI Agent（需要 ANTHROPIC_API_KEY）
    if os.getenv("ANTHROPIC_API_KEY"):
        test_agent()
    else:
        print("\n[跳过测试二] 请先设置 ANTHROPIC_API_KEY 环境变量")
        print("  Windows: set ANTHROPIC_API_KEY=sk-xxx")
        print("  Linux/Mac: export ANTHROPIC_API_KEY=sk-xxx")
