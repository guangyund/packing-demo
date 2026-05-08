"""
本地3D可视化 - 使用 matplotlib 渲染装箱结果
用法：
    from visualizer import visualize_packing
    visualize_packing(packing_result)          # 直接弹窗显示
    visualize_packing(packing_result, "out.png")  # 保存为图片
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 货物颜色列表
COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57",
    "#FF9FF3", "#54A0FF", "#A29BFE", "#00D2D3", "#FF9F43",
    "#6C5CE7", "#FD79A8", "#55EFC4", "#FDCB6E", "#E17055",
]


def _draw_box(ax, x, y, z, dx, dy, dz, color, alpha=0.55):
    """在3D坐标系中绘制一个实心箱体"""
    # 8个顶点（x=长, y=宽, z=高）
    verts = [
        (x,    y,    z),    (x+dx, y,    z),
        (x+dx, y+dy, z),    (x,    y+dy, z),
        (x,    y,    z+dz), (x+dx, y,    z+dz),
        (x+dx, y+dy, z+dz), (x,    y+dy, z+dz),
    ]
    # 6个面
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],  # 底面
        [verts[4], verts[5], verts[6], verts[7]],  # 顶面
        [verts[0], verts[1], verts[5], verts[4]],  # 前面
        [verts[2], verts[3], verts[7], verts[6]],  # 后面
        [verts[0], verts[3], verts[7], verts[4]],  # 左面
        [verts[1], verts[2], verts[6], verts[5]],  # 右面
    ]
    poly = Poly3DCollection(
        faces, alpha=alpha,
        facecolor=color, edgecolor="black", linewidth=0.4,
    )
    ax.add_collection3d(poly)


def _draw_bin_wireframe(ax, L, W, H):
    """绘制箱体外框（虚线）"""
    corners = [
        (0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0),
        (0, 0, H), (L, 0, H), (L, W, H), (0, W, H),
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # 底面
        (4,5),(5,6),(6,7),(7,4),   # 顶面
        (0,4),(1,5),(2,6),(3,7),   # 四条竖边
    ]
    for a, b in edges:
        xs = [corners[a][0], corners[b][0]]
        ys = [corners[a][1], corners[b][1]]
        zs = [corners[a][2], corners[b][2]]
        ax.plot3D(xs, ys, zs, "k--", alpha=0.3, linewidth=1)


def visualize_packing(packing_result: dict, save_path: str = None):
    """
    可视化装箱结果

    Args:
        packing_result: calculate_packing() 或 agent 返回的最终装箱结果
        save_path:      图片保存路径（None 则直接弹窗显示）
    """
    packed_bins = packing_result.get("packed_bins", [])
    if not packed_bins:
        print("[可视化] 没有装箱结果可显示")
        return

    n = len(packed_bins)
    fig = plt.figure(figsize=(min(8 * n, 20), 7), facecolor="#1a1a2e")
    fig.suptitle("三维装箱可视化", fontsize=14, fontweight="bold", color="white")

    for idx, bin_data in enumerate(packed_bins):
        ax = fig.add_subplot(1, n, idx + 1, projection="3d", facecolor="#16213e")

        dim = bin_data["dimensions"]
        L, W, H = dim["length"], dim["width"], dim["height"]

        # 绘制箱体外框
        _draw_bin_wireframe(ax, L, W, H)

        # 绘制货物 + 收集图例
        patches = []
        for i, item in enumerate(bin_data["items"]):
            color = COLORS[i % len(COLORS)]
            p = item["position"]
            d = item["dimensions"]
            _draw_box(ax, p["x"], p["y"], p["z"], d["length"], d["width"], d["height"], color)
            patches.append(mpatches.Patch(color=color, label=item["id"]))

        ax.set_xlim(0, L)
        ax.set_ylim(0, W)
        ax.set_zlim(0, H)
        ax.set_xlabel("长 (cm)", color="white", fontsize=8)
        ax.set_ylabel("宽 (cm)", color="white", fontsize=8)
        ax.set_zlabel("高 (cm)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        title = (
            f"箱子{idx+1}：{bin_data['bin_type']}\n"
            f"利用率 {bin_data['utilization']*100:.1f}%  |  {bin_data['item_count']} 件"
        )
        ax.set_title(title, color="white", fontsize=10, pad=10)
        ax.legend(
            handles=patches, fontsize=7, loc="upper left",
            bbox_to_anchor=(1.02, 1), framealpha=0.2,
            labelcolor="white", facecolor="#0f3460",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[可视化] 图片已保存：{save_path}")
    else:
        plt.show()
