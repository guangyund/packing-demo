#!/bin/bash
# ============================================================
# packing_demo 回滚脚本
# 用法：
#   ./rollback.sh         # 交互式选择回滚版本
#   ./rollback.sh v1.0.0  # 直接回滚到指定 tag
# ============================================================
set -e

PROJECT_DIR="/opt/packing_demo"
SERVICE_NAME="packing"

cd "$PROJECT_DIR"
git fetch --tags origin

CURRENT=$(git describe --tags --abbrev=0 2>/dev/null || echo "未知")
echo "当前版本：$CURRENT"
echo ""
echo "可用历史版本："
git tag --sort=-version:refname | head -10

# 确定回滚目标
if [ -n "$1" ]; then
    TARGET="$1"
else
    echo ""
    read -p "请输入要回滚到的版本（如 v1.0.0）：" TARGET
fi

if [ -z "$TARGET" ]; then
    echo "❌ 未指定版本，取消回滚"
    exit 1
fi

echo ""
echo "⚠️  即将从 $CURRENT 回滚到 $TARGET，是否继续？(y/N)"
read -r CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "已取消"
    exit 0
fi

# ── 切换版本 ────────────────────────────────────────────────
git checkout "$TARGET"

# ── 还原依赖 ────────────────────────────────────────────────
echo "📥 还原 Python 依赖..."
venv/bin/pip install -r requirements.txt -q

# ── 重启服务 ────────────────────────────────────────────────
echo "🔄 重启服务 $SERVICE_NAME ..."
sudo systemctl restart "$SERVICE_NAME"
sleep 2
sudo systemctl status "$SERVICE_NAME" --no-pager -l

echo ""
echo "✅ 已回滚到：$TARGET"
