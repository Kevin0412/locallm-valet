#!/usr/bin/env bash
# Clash Verge 用 TCP 33331 做单实例锁（硬编码）。SGLang 的 scheduler 子进程
# 使用内核随机分配的临时端口（bind(0)），可能撞上 33331，导致 Clash GUI
# 静默无法启动。本脚本把 33331 从临时端口范围永久保留，杜绝复发。
#
# Usage: bash deploy/fix-clash-verge-port.sh   （需要 sudo 密码）
set -euo pipefail

echo 'net.ipv4.ip_local_reserved_ports=33331' | sudo tee /etc/sysctl.d/99-clash-verge-port.conf >/dev/null
sudo sysctl --system >/dev/null
echo "[1/2] 33331 已从内核临时端口范围保留"

echo "[2/2] 重启 locallm-valet（释放可能占着 33331 的旧 sglang 临时端口）..."
sudo systemctl restart locallm-valet
echo "完成。现在可以正常启动 clash-verge；以后 sglang 重启也抢不到 33331。"
