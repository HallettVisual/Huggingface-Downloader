@echo off
title Hugging Face Model Downloader
python "%~dp0hf_downloader.py" %*
if errorlevel 1 (
    echo.
    echo The downloader exited with an error. Press a key to close.
    pause >nul
)
