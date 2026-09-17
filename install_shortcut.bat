@echo off
rem Double-click to add HF Model Downloader to the Start menu and desktop.
rem Execution policy is bypassed for this one script run only.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_shortcut.ps1" -Desktop %*
echo.
pause
