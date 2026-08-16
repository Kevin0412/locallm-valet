@echo off
rem llm-gateway Windows launcher example (or register with NSSM as a service)
cd /d %~dp0..
set PYTHONPATH=%CD%\.deps;%CD%
python -m llm_gateway --config config.yaml
