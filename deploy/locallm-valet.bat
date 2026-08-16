@echo off
rem locallm-valet Windows launcher example (or register with NSSM as a service)
cd /d %~dp0..
set PYTHONPATH=%CD%\.deps;%CD%
python -m locallm_valet --config config.yaml
