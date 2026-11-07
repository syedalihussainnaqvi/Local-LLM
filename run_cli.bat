@echo off
title Qwen 2.5 LoRA Console
echo ========================================================
echo       Starting Qwen 2.5 LoRA Console Interface
echo ========================================================
where py >nul 2>nul
if %errorlevel% equ 0 (
    py cli_chat.py
) else (
    python cli_chat.py
)
pause
