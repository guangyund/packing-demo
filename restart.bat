@echo off
chcp 65001 >nul
echo [重启] 正在停止所有 Python 进程...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul
echo [重启] 启动 packing_demo 服务...
start "packing_demo" cmd /k "cd /d %~dp0 && python main.py"
echo [重启] 完成，服务已在新窗口启动
