#!/bin/bash
# ============================================================
# packing_demo 部署脚本
# 用法：
#   ./deploy.sh           # 部署最新 tag
#   ./deploy.sh v1.0.2    # 部署指定 tag
# ============================================================
set -e

PROJECT_DIR="/opt/packing_demo"
SERVICE_NAME="packing"          # systemctl 服务名，按实际改

cd "$PROJECT_DIR"

# ── 拉取最新代码和标签 ──────────────────────────────────────
echo "📦 拉取最新代码..."
git fetch --tags origin

# 确定目标版本
if [ -n "$1" ]; then
    TARGET="$1"
else
    TARGET=$(git tag --sort=-version:refname | head -1)
fi

CURRENT=$(git describe --tags --abbrev=0 2>/dev/null || echo "无")
echo "当前版本：$CURRENT  →  目标版本：$TARGET"

# ── 切换版本 ────────────────────────────────────────────────
git checkout "$TARGET"

# ── 安装/更新依赖 ────────────────────────────────────────────
echo "📥 更新 Python 依赖..."
venv/bin/pip install -r requirements.txt -q

# ── 重启服务 ────────────────────────────────────────────────
echo "🔄 重启服务 $SERVICE_NAME ..."
sudo systemctl restart "$SERVICE_NAME"
sleep 2
sudo systemctl status "$SERVICE_NAME" --no-pager -l

echo ""
echo "✅ 部署完成：$TARGET"
