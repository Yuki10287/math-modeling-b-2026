@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0src\q4_model\run_share25.ps1"
exit /b %errorlevel%
