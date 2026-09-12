@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_share25.ps1"
exit /b %errorlevel%
