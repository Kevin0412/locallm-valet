@echo off
rem llm-gateway Windows 启动示例（或使用 NSSM 注册为服务）
cd /d %~dp0..
set PYTHONPATH=%CD%\.deps;%CD%
python -m llm_gateway --config config.yaml
