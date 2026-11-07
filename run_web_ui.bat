@echo off
title Qwen 2.5 LoRA Studio
echo ========================================================
echo       Starting Qwen 2.5 LoRA Studio (Web Interface)
echo ========================================================
echo Opening browser to http://127.0.0.1:5000 ...
start http://127.0.0.1:5000
where py >nul 2>nul
if %errorlevel% equ 0 (
    py app.py
) else (
    python app.py
)
pause
