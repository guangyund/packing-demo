"""
三维装箱核心算法 - 贪心热启动 + OR-Tools CP-SAT + 多线程

执行流程：
  1. 角点贪心算法 → 毫秒级算出初始可行解（含旋转选择）
  2. 将贪心结果作为 Hint 传入 OR-Tools（固定旋转，只优化位置）
  3. OR-Tools 多线程并行搜索更优方案
  4. 若 OR-Tools 超时未找到更好解，直接返回贪心结果（保证有解）
"""
from ortools.sat.python import cp_model
from config import (MAX_BIN_WEIGHT, ENABLE_GRAVITY_SETTLING, ORTOOLS_TIME_LIMIT, ORTOOLS_NUM_WORKERS,
                    ENABLE_SUPPORT_CHECK, MIN_ITEM_SUPPORT_RATIO)

# float → int 转换精度（0.1cm 精度）
SCALE = 10

# 6 种旋转方向索引：(长轴, 宽轴, 高轴)
ROTATIONS = [
    (0, 1, 2),  # 原始方向
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]


def _i(val: float) -> int:
    return int(round(val * SCALE))


# ── 第一步：角点贪心算法 ──────────────────────────────────────────────────────

def _dims(item: dict, rot: tuple) -> tuple:
    d = [item["length"], item["width"], item["height"]]
    return d[rot[0]], d[rot[1]], d[rot[2]]


def _overlaps(x1, y1, z1, dx1, dy1, dz1, x2, y2, z2, dx2, dy2, dz2) -> bool:
    return not (
        x1 + dx1 <= x2 or x2 + dx2 <= x1 or
        y1 + dy1 <= y2 or y2 + dy2 <= y1 or
        z1 + dz1 <= z2 or z2 + dz2 <= z1
    )


def _support_ratio(x: float, y: float, z: float,
                   dl: float, dw: float,
                   placed_raw: list, eps: float = 1e-6) -> float:
    """
    计算货物底面在高度 z 处的支撑面积比。
    z=0 视为地面全面积支撑，返回 1.0。
    """
    if z < eps:
        return 1.0
    item_area = dl * dw
    if item_area < eps:
        return 1.0
    supported = 0.0
    for px, py, pz, pdl, pdw, pdh in placed_raw:
        if abs((pz + pdh) - z) < eps:          # 顶面恰好在高度 z 处
            ox = min(x + dl, px + pdl) - max(x, px)
            oy = min(y + dw, py + pdw) - max(y, py)
            if ox > eps and oy > eps:
                supported += ox * oy
    return supported / item_area


def _greedy_pack(items: list, bin_data: dict, preferred_rot: tuple = None) -> tuple:
    """
    角点贪心算法：毫秒级快速装箱，用于生成热启动初始解
    货物按体积从大到小排序，优先安排大件，减少碎片化空间。
    preferred_rot: 相同位置得分时优先尝试的旋转方向，用于多策略搜索
    """
    items = sorted(items, key=lambda x: x["length"] * x["width"] * x["height"], reverse=True)
    BL, BW, BH   = bin_data["length"], bin_data["width"], bin_data["height"]
    max_weight    = bin_data["max_weight"]
    placed_raw    = []   # (x, y, z, dl, dw, dh)
    current_weight = 0.0
    corners       = [(0.0, 0.0, 0.0)]
    seen_corners  = {(0.0, 0.0, 0.0)}
    placed_items  = []
    unplaced_items = []
    EPS = 1e-6

    # 将优先旋转方向移到首位，相同得分时它会赢得决胜
    rotations = list(ROTATIONS)
    if preferred_rot is not None and preferred_rot in rotations:
        rotations.remove(preferred_rot)
        rotations.insert(0, preferred_rot)

    for item in items:
        if current_weight + item["weight"] > max_weight + EPS:
            unplaced_items.append(item)
            continue

        best_pos   = None
        best_rot   = None
        best_score = float("inf")

        for x, y, z in corners:
            for rot in rotations:
                dl, dw, dh = _dims(item, rot)
                if x + dl > BL + EPS or y + dw > BW + EPS or z + dh > BH + EPS:
                    continue
                overlap = any(
                    _overlaps(x, y, z, dl, dw, dh, p[0], p[1], p[2], p[3], p[4], p[5])
                    for p in placed_raw
                )
                if not overlap:
                    # 支撑率检查：货物底面必须有足够比例被支撑
                    if (ENABLE_SUPPORT_CHECK and z > EPS and
                            _support_ratio(x, y, z, dl, dw, placed_raw) < MIN_ITEM_SUPPORT_RATIO):
                        continue
                    # dh * 1e4：惩罚竖放（高度大的摆向），优先选矮的旋转方向
                    score = z * 1e8 + y * 1e4 + x + dh * 1e4
                    if score < best_score:
                        best_score = score
                        best_pos   = (x, y, z)
                        best_rot   = rot

        if best_pos:
            x, y, z    = best_pos
            dl, dw, dh = _dims(item, best_rot)
            placed_raw.append((x, y, z, dl, dw, dh))
            current_weight += item["weight"]
            for nc in [(x + dl, y, z), (x, y + dw, z), (x, y, z + dh)]:
                if nc not in seen_corners:
                    seen_corners.add(nc)
                    corners.append(nc)
            placed_items.append({
                "id": item["id"],
                "position":   {"x": float(x),  "y": float(y),  "z": float(z)},
                "dimensions": {"length": float(dl), "width": float(dw), "height": float(dh)},
                "rotation_type": ROTATIONS.index(best_rot),
            })
        else:
            unplaced_items.append(item)

    return placed_items, unplaced_items


# ── 第二步：OR-Tools 精确优化（固定旋转，只优化位置）────────────────────────

def _ortools_optimize(items: list, bin_data: dict, greedy_placed: list,
                      time_limit: float, num_workers: int) -> tuple:
    """
    在贪心结果基础上，用 OR-Tools 进一步优化摆放位置
    固定旋转方向（来自贪心），只优化 x/y/z 坐标，大幅降低模型复杂度
    """
    model = cp_model.CpModel()

    BL = _i(bin_data["length"])
    BW = _i(bin_data["width"])
    BH = _i(bin_data["height"])
    n  = len(items)

    # 使用贪心确定的旋转方向作为固定尺寸
    greedy_map = {p["id"]: p for p in greedy_placed}
    item_dims  = []
    for item in items:
        if item["id"] in greedy_map:
            d = greedy_map[item["id"]]["dimensions"]
            item_dims.append((_i(d["length"]), _i(d["width"]), _i(d["height"])))
        else:
            item_dims.append((_i(item["length"]), _i(item["width"]), _i(item["height"])))

    # ── 决策变量 ──────────────────────────────────────────────────────────────
    x      = [model.new_int_var(0, BL, f"x{i}") for i in range(n)]
    y      = [model.new_int_var(0, BW, f"y{i}") for i in range(n)]
    z      = [model.new_int_var(0, BH, f"z{i}") for i in range(n)]
    placed = [model.new_bool_var(f"p{i}")        for i in range(n)]

    for i in range(n):
        dl, dw, dh = item_dims[i]

        # 边界约束
        model.add(x[i] + dl <= BL).only_enforce_if(placed[i])
        model.add(y[i] + dw <= BW).only_enforce_if(placed[i])
        model.add(z[i] + dh <= BH).only_enforce_if(placed[i])
        model.add(x[i] == 0).only_enforce_if(placed[i].negated())
        model.add(y[i] == 0).only_enforce_if(placed[i].negated())
        model.add(z[i] == 0).only_enforce_if(placed[i].negated())
        if dl > BL or dw > BW or dh > BH:
            model.add(placed[i] == 0)

    # 不重叠约束
    for i in range(n):
        for j in range(i + 1, n):
            dl_i, dw_i, dh_i = item_dims[i]
            dl_j, dw_j, dh_j = item_dims[j]
            sep = [model.new_bool_var(f"s{i}_{j}_{k}") for k in range(6)]
            model.add(x[i] + dl_i <= x[j]).only_enforce_if(sep[0])
            model.add(x[j] + dl_j <= x[i]).only_enforce_if(sep[1])
            model.add(y[i] + dw_i <= y[j]).only_enforce_if(sep[2])
            model.add(y[j] + dw_j <= y[i]).only_enforce_if(sep[3])
            model.add(z[i] + dh_i <= z[j]).only_enforce_if(sep[4])
            model.add(z[j] + dh_j <= z[i]).only_enforce_if(sep[5])
            model.add_bool_or([placed[i].negated(), placed[j].negated()] + sep)

    # 重量约束
    weights_int    = [_i(item["weight"]) for item in items]
    max_weight_int = _i(bin_data["max_weight"])
    model.add(sum(placed[i] * weights_int[i] for i in range(n)) <= max_weight_int)

    # 目标：最大化放入货物总体积
    volumes = [item_dims[i][0] * item_dims[i][1] * item_dims[i][2] for i in range(n)]
    model.maximize(sum(placed[i] * volumes[i] for i in range(n)))

    # ── 热启动：注入贪心结果作为初始 Hint ─────────────────────────────────────
    for i, item in enumerate(items):
        if item["id"] in greedy_map:
            p = greedy_map[item["id"]]
            model.add_hint(x[i], _i(p["position"]["x"]))
            model.add_hint(y[i], _i(p["position"]["y"]))
            model.add_hint(z[i], _i(p["position"]["z"]))
            model.add_hint(placed[i], 1)
        else:
            model.add_hint(placed[i], 0)

    # ── 多线程求解 ─────────────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers  = num_workers  # 多线程并行
    status = solver.solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return None  # 未找到解，调用方回退到贪心结果

    placed_items, unplaced_items = [], []
    for i, item in enumerate(items):
        if solver.value(placed[i]):
            dl, dw, dh = item_dims[i]
            placed_items.append({
                "id": item["id"],
                "position": {
                    "x": solver.value(x[i]) / SCALE,
                    "y": solver.value(y[i]) / SCALE,
                    "z": solver.value(z[i]) / SCALE,
                },
                "dimensions": {
                    "length": dl / SCALE,
                    "width":  dw / SCALE,
                    "height": dh / SCALE,
                },
                "rotation_type": greedy_map[item["id"]]["rotation_type"] if item["id"] in greedy_map else 0,
            })
        else:
            unplaced_items.append(item)

    return placed_items, unplaced_items


# ── 重力沉降：消除浮空 ────────────────────────────────────────────────────────

def _settle_gravity(placed_items: list) -> list:
    """
    模拟重力，将所有货物向下沉降到最低合法位置，消除浮空。

    当 ENABLE_SUPPORT_CHECK=True 时，额外检查底面支撑率：
      - 从最高候选落点开始，依次尝试更低的候选落点
      - 找到第一个"支撑率 >= MIN_ITEM_SUPPORT_RATIO 且无碰撞"的高度即停止
      - 若所有候选均不满足（通常因低位已被其他货物占据），退回原始逻辑（最高落点）
    """
    if not placed_items:
        return placed_items

    EPS = 1e-6
    sorted_items = sorted(placed_items, key=lambda p: p["position"]["z"])
    settled_raw = []   # (x, y, z, dl, dw, dh) 已沉降货物
    result = []

    for item in sorted_items:
        p  = item["position"]
        d  = item["dimensions"]
        x, y       = p["x"], p["y"]
        dl, dw, dh = d["length"], d["width"], d["height"]
        item_area  = dl * dw

        # ── 收集所有候选落点：地面(0) + 与当前货物水平重叠的已沉降货物顶面 ──────
        candidate_z: set = {0.0}
        for sx, sy, sz, sdl, sdw, sdh in settled_raw:
            ox = min(x + dl, sx + sdl) - max(x, sx)
            oy = min(y + dw, sy + sdw) - max(y, sy)
            if ox > EPS and oy > EPS:
                candidate_z.add(sz + sdh)

        if not ENABLE_SUPPORT_CHECK:
            # 原始逻辑：直接取最高候选落点
            new_z = max(candidate_z)
        else:
            # 支撑率检查：从最高候选落点向下找第一个满足条件的高度
            new_z = None
            for cz in sorted(candidate_z, reverse=True):
                # ① 地面永远满足支撑率（地面全面积支撑）
                if cz == 0.0:
                    support_ok = True
                else:
                    # 统计在高度 cz 处与货物底面重叠的支撑面积
                    supported = 0.0
                    for sx, sy, sz, sdl, sdw, sdh in settled_raw:
                        if abs((sz + sdh) - cz) < EPS:   # 顶面恰好在 cz
                            ox = min(x + dl, sx + sdl) - max(x, sx)
                            oy = min(y + dw, sy + sdw) - max(y, sy)
                            if ox > EPS and oy > EPS:
                                supported += ox * oy
                    support_ok = (item_area < EPS) or (supported / item_area >= MIN_ITEM_SUPPORT_RATIO)

                if not support_ok:
                    continue    # 支撑不足，尝试更低候选

                # ② 检查在 [cz, cz+dh] 范围内无碰撞（不与已沉降货物重叠）
                collision = False
                for sx, sy, sz, sdl, sdw, sdh in settled_raw:
                    ox = min(x + dl, sx + sdl) - max(x, sx)
                    oy = min(y + dw, sy + sdw) - max(y, sy)
                    if ox <= EPS or oy <= EPS:
                        continue
                    oz = min(cz + dh, sz + sdh) - max(cz, sz)
                    if oz > EPS:
                        collision = True
                        break

                if not collision:
                    new_z = cz
                    break   # 找到最高有效支撑位，停止

            if new_z is None:
                # 所有候选均不满足（低位被占）→ 退回原始最高落点，接受架桥
                new_z = max(candidate_z)

        settled_raw.append((x, y, new_z, dl, dw, dh))
        result.append({
            **item,
            "position": {"x": x, "y": y, "z": new_z},
        })

    return result


# ── 水平压缩：消除 x/y 方向间隙 ───────────────────────────────────────────────

def _settle_horizontal(placed_items: list) -> list:
    """
    模拟"靠墙"压缩：先沿 y 轴向 y=0 滑动，再沿 x 轴向 x=0 滑动，
    消除贪心 / OR-Tools 在水平方向产生的间隙。
    逻辑与 _settle_gravity 完全对称，只是压缩方向改为水平。
    """
    if not placed_items:
        return placed_items

    EPS = 1e-6

    # ── 第一步：向 y=0 压缩 ──────────────────────────────────────────────────
    cur = sorted(placed_items, key=lambda p: p["position"]["y"])
    raw, out = [], []
    for item in cur:
        p, d = item["position"], item["dimensions"]
        x, z = p["x"], p["z"]
        dl, dw, dh = d["length"], d["width"], d["height"]
        ny = 0.0
        for sx, sy, sz, sdl, sdw, sdh in raw:
            if (min(x+dl, sx+sdl) - max(x, sx) > EPS and
                    min(z+dh, sz+sdh) - max(z, sz) > EPS):
                ny = max(ny, sy + sdw)
        raw.append((x, ny, z, dl, dw, dh))
        out.append({**item, "position": {**p, "y": ny}})

    # ── 第二步：向 x=0 压缩 ──────────────────────────────────────────────────
    cur = sorted(out, key=lambda p: p["position"]["x"])
    raw, out = [], []
    for item in cur:
        p, d = item["position"], item["dimensions"]
        y, z = p["y"], p["z"]
        dl, dw, dh = d["length"], d["width"], d["height"]
        nx = 0.0
        for sx, sy, sz, sdl, sdw, sdh in raw:
            if (min(y+dw, sy+sdw) - max(y, sy) > EPS and
                    min(z+dh, sz+sdh) - max(z, sz) > EPS):
                nx = max(nx, sx + sdl)
        raw.append((nx, y, z, dl, dw, dh))
        out.append({**item, "position": {**p, "x": nx}})

    return out


# ── 层排序：底部面积最大 ───────────────────────────────────────────────────────

def _sort_layers_by_coverage(placed_items: list, bin_height: float = None) -> list:
    """
    将装箱结果按层重新排序，使底部层的水平占用面积最大、往上依次减小。
    纯后处理，不改变货物数量，只调整层的 z 顺序。

    安全约束：当箱内存在多列布局（各区域堆叠高度不同），所有层 max_h 之和可能
    超过箱高（例如两列并排，高列 5 层×6.5cm + 矮列 2 层×12cm = 56.5cm > 箱高）。
    此时线性重排会导致货物飞出箱外，直接返回原坐标不做排序。
    """
    if len(placed_items) <= 1:
        return placed_items

    LAYER_TOL = 1.0  # z 相差 ≤1cm 视为同一层

    # 按 z 分层
    sorted_by_z = sorted(placed_items, key=lambda p: p["position"]["z"])
    layers = []
    for item in sorted_by_z:
        z = item["position"]["z"]
        if layers and z - layers[-1]["floor"] <= LAYER_TOL:
            layers[-1]["items"].append(item)
            layers[-1]["max_h"] = max(layers[-1]["max_h"], item["dimensions"]["height"])
        else:
            layers.append({
                "floor": z,
                "max_h": item["dimensions"]["height"],
                "items": [item],
            })

    # 计算每层水平占用面积（各货物 footprint 之和）
    for layer in layers:
        layer["area"] = sum(
            it["dimensions"]["length"] * it["dimensions"]["width"]
            for it in layer["items"]
        )

    # 安全检查：所有层 max_h 之和 > 箱高，说明是多列布局，不能线性重排
    total_stacked_h = sum(layer["max_h"] for layer in layers)
    if bin_height is not None and total_stacked_h > bin_height + 1e-6:
        return placed_items  # 放弃排序，保留原坐标

    # 面积从大到小排序（底部面积最大）
    layers.sort(key=lambda l: l["area"], reverse=True)

    # 重新分配 z 坐标
    result = []
    current_z = 0.0
    for layer in layers:
        for item in layer["items"]:
            result.append({**item, "position": {**item["position"], "z": current_z}})
        current_z += layer["max_h"]

    return result


# ── 主装箱函数 ────────────────────────────────────────────────────────────────

def _pack_bin(items: list, bin_data: dict,
              time_limit: float = ORTOOLS_TIME_LIMIT,
              num_workers: int  = ORTOOLS_NUM_WORKERS,
              scan_mode: bool   = False) -> tuple:
    """
    贪心热启动 + OR-Tools + 多线程

    1. 贪心快速得到初始解（保底）
    2. OR-Tools 在初始解基础上优化
    3. 若 OR-Tools 超时无解，返回贪心结果
    """
    # Step 1：多策略贪心 —— 用 6 种旋转方向各跑一次，取装入件数最多的结果
    # 解决单一方向贪心错过最优层布局的问题（如同规格货物换面装能多放一层）
    greedy_placed, greedy_unplaced = [], items[:]
    for pref_rot in ROTATIONS:
        g_placed, g_unplaced = _greedy_pack(items, bin_data, preferred_rot=pref_rot)
        if len(g_placed) > len(greedy_placed):
            greedy_placed   = g_placed
            greedy_unplaced = g_unplaced

    if not greedy_placed:
        return [], items

    def _settle(placed):
        settled = _settle_gravity(placed) if ENABLE_GRAVITY_SETTLING else placed
        settled = _settle_horizontal(settled)
        return _sort_layers_by_coverage(settled, bin_height=bin_data["height"])

    # 贪心已装完所有货物时，直接返回贪心结果
    # OR-Tools 此时只会重排位置，反而可能打乱整齐的分层结构，无需再跑
    if not greedy_unplaced:
        return _settle(greedy_placed), []

    # Step 2：OR-Tools 优化（仅在贪心无法装完全部货物时运行；scan_mode 下跳过）
    if scan_mode:
        return _settle(greedy_placed), greedy_unplaced

    ortools_result = _ortools_optimize(
        items, bin_data, greedy_placed, time_limit, num_workers
    )

    if ortools_result is None:
        # OR-Tools 未找到解，回退到贪心结果
        return _settle(greedy_placed), greedy_unplaced

    ortools_placed, ortools_unplaced = ortools_result

    # 比较两种结果，取放入货物数更多的
    if len(ortools_placed) >= len(greedy_placed):
        return _settle(ortools_placed), ortools_unplaced
    else:
        return _settle(greedy_placed), greedy_unplaced


def _greedy_all_fit(items: list, bin_data: dict) -> bool:
    """
    用贪心算法快速判断所有货物能否装入单箱（毫秒级，不运行 OR-Tools）。
    仅用于单箱可行性筛选，不做最终坐标输出。
    """
    _, unplaced = _greedy_pack(items, bin_data)
    return len(unplaced) == 0


def _make_bin_record(bin_data: dict, placed: list, weight_map: dict) -> dict:
    """构造单箱结果记录（复用逻辑）"""
    bin_volume   = bin_data["length"] * bin_data["width"] * bin_data["height"]
    total_weight = sum(weight_map[p["id"]] for p in placed)

    # 利用率 = 货物体积之和 / 箱子体积（真实填充率）
    item_volume = sum(
        p["dimensions"]["length"] * p["dimensions"]["width"] * p["dimensions"]["height"]
        for p in placed
    ) if placed else 0.0

    return {
        "bin_type":    bin_data["type"],
        "dimensions":  {
            "length": float(bin_data["length"]),
            "width":  float(bin_data["width"]),
            "height": float(bin_data["height"]),
        },
        "utilization":  round(item_volume / bin_volume, 2) if bin_volume > 0 else 0.0,
        "total_weight": round(total_weight, 2),
        "item_count":   len(placed),
        "items":        placed,
    }


def calculate_packing(items: list, bins: list, scan_mode: bool = False) -> dict:
    """
    执行三维装箱计算

    Args:
        items: 货物列表 [{"id","length","width","height","weight"}]
        bins:  箱型列表 [{"type","length","width","height","max_weight"}]

    策略：
        1. 单箱优先：从小到大尝试每种箱型，找到能装下所有货物的最小单箱即返回
        2. 多箱兜底：无法单箱装完时，贪心多箱（小箱优先，大件先装）
    """
    remaining = sorted(
        items,
        key=lambda x: x["length"] * x["width"] * x["height"],
        reverse=True,
    )

    sorted_bins = sorted(
        bins,
        key=lambda b: b["length"] * b["width"] * b["height"],
    )

    weight_map = {item["id"]: item["weight"] for item in items}

    # ── 第一步：单箱优先 ────────────────────────────────────────────────────────
    # 用「体积/重量/单件可放」三项快速过滤，对每个通过的箱型跑完整装箱（贪心+OR-Tools）。
    # 不再用贪心做预筛——贪心单一方向会漏掉「混合旋转」才能装满的布局，导致错误排除。
    #
    # 候选选择规则：
    #   · 有利用率 ≥80% 的 → 取体积最小那个（sorted_bins 已按体积升序，遇到即 break）
    #   · 全部 <80%        → 取利用率最高的
    total_item_volume = sum(
        item["length"] * item["width"] * item["height"] for item in remaining
    )
    total_weight = sum(item["weight"] for item in remaining)

    single_box_results = []   # [(bin_data, placed_items, utilization)]
    for bin_data in sorted_bins:
        effective_bin = dict(bin_data)
        effective_bin["max_weight"] = min(bin_data["max_weight"], MAX_BIN_WEIGHT)
        bin_volume = bin_data["length"] * bin_data["width"] * bin_data["height"]

        # 快速过滤 ①：总重超限
        if total_weight > effective_bin["max_weight"]:
            continue

        # 快速过滤 ②：货物总体积超过箱子体积
        if total_item_volume > bin_volume:
            continue

        # 快速过滤 ③：存在某件货物六种旋转都放不进箱子
        if not all(
            any(
                _dims(item, rot)[0] <= bin_data["length"] and
                _dims(item, rot)[1] <= bin_data["width"]  and
                _dims(item, rot)[2] <= bin_data["height"]
                for rot in ROTATIONS
            )
            for item in remaining
        ):
            continue

        # 完整装箱：多策略贪心 + OR-Tools
        placed, unplaced = _pack_bin(remaining, effective_bin, scan_mode=scan_mode)
        if unplaced:
            continue  # 装不完，不纳入候选

        used_volume = sum(
            p["dimensions"]["length"] * p["dimensions"]["width"] * p["dimensions"]["height"]
            for p in placed
        )
        util = used_volume / bin_volume
        single_box_results.append((bin_data, placed, util))

        if util >= 0.80:
            break  # 找到利用率达标的最小箱型，无需继续

    if single_box_results:
        high_util = [(b, p, u) for b, p, u in single_box_results if u >= 0.80]
        chosen_bin_data, chosen_placed, _ = (
            high_util[0]
            if high_util
            else max(single_box_results, key=lambda x: x[2])
        )
        record = _make_bin_record(chosen_bin_data, chosen_placed, weight_map)
        return {
            "packed_bins":    [record],
            "unplaced_items": [],
            "summary": {
                "total_bins_used": 1,
                "avg_utilization": record["utilization"],
                "all_placed":      True,
            },
        }

    # ── 第二步：无法单箱，每轮为剩余货物独立选出最优箱型 ────────────────────
    packed_bins = []

    while remaining:
        best_bin_data     = None
        best_placed_count = 0
        best_util         = -1.0

        for bin_data in sorted_bins:
            effective_bin = dict(bin_data)
            effective_bin["max_weight"] = min(bin_data["max_weight"], MAX_BIN_WEIGHT)

            # 多策略贪心评估（6种旋转方向各跑一次，取装入件数最多的结果）
            # 避免单方向贪心因错过混合旋转布局而低估小箱型的装载能力
            best_g_placed = []
            for pref_rot in ROTATIONS:
                g_placed, _ = _greedy_pack(remaining, effective_bin, preferred_rot=pref_rot)
                if len(g_placed) > len(best_g_placed):
                    best_g_placed = g_placed

            if not best_g_placed:
                continue

            bin_volume  = bin_data["length"] * bin_data["width"] * bin_data["height"]
            used_volume = sum(
                p["dimensions"]["length"] * p["dimensions"]["width"] * p["dimensions"]["height"]
                for p in best_g_placed
            )
            util = used_volume / bin_volume
            cnt  = len(best_g_placed)

            # 选箱优先级：① 装入件数多 → ② 空间利用率高
            if (cnt, util) > (best_placed_count, best_util):
                best_placed_count = cnt
                best_util         = util
                best_bin_data     = bin_data

        if not best_bin_data:
            break  # 所有箱型都装不下剩余货物，终止

        effective_bin = dict(best_bin_data)
        effective_bin["max_weight"] = min(best_bin_data["max_weight"], MAX_BIN_WEIGHT)
        placed, remaining = _pack_bin(remaining, effective_bin, scan_mode=scan_mode)
        if not placed:
            break
        packed_bins.append(_make_bin_record(best_bin_data, placed, weight_map))

    unplaced_ids = [item["id"] for item in remaining]

    return {
        "packed_bins":    packed_bins,
        "unplaced_items": unplaced_ids,
        "summary": {
            "total_bins_used": len(packed_bins),
            "avg_utilization": round(
                sum(b["utilization"] for b in packed_bins) / len(packed_bins), 2
            ) if packed_bins else 0,
            "all_placed": len(unplaced_ids) == 0,
        },
    }


# 仓库可用箱型
AVAILABLE_BINS = [
    {"type": "小号箱", "length": 40, "width": 30, "height": 30, "max_weight": 15},
    {"type": "中号箱", "length": 60, "width": 50, "height": 50, "max_weight": 30},
    {"type": "大号箱", "length": 80, "width": 60, "height": 60, "max_weight": 50},
    {"type": "超大箱", "length": 100, "width": 80, "height": 80, "max_weight": 80},
]
